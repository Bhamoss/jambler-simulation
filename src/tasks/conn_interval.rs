use itertools::Itertools;
use plotters::prelude::*;
use rand::{Rng, seq::SliceRandom};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{fs::{create_dir_all, File}};

use statrs::distribution::{Binomial,Geometric};
use statrs::{distribution::Univariate, statistics::Mean};
use std::f64;
use num::{Integer, integer::{binomial, gcd}};

use crate::csa2::BlePhy;
use crate::{SimulationParameters, Task, run_tasks, tasks::BleConnection};


use std::sync::{Arc, Mutex};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};


// IMPORTANT does not take into account the possible 16 and distance drift. This is much too restrictive and cripples this approach
// IMPORTANT from my understanding this is software time. Mention this in thesis
const ROUND_THRESS : u64 = (625f64 * (1000000f64 / (500f64+20f64))) as u64;
pub fn conn_interval<R: RngCore + Send + Sync>(mut params: SimulationParameters<R>, bars: Arc<Mutex<MultiProgress>>) {
    params.output_dir.push("conn_interval");
    create_dir_all(&params.output_dir).unwrap();
    let tasks: Vec<Box<dyn Task>> = vec![
        
        Box::new(gcd_sim),
        Box::new(too_much_drift),
        Box::new(bates_cdf_plot),
        Box::new(bates_necessary_n),
        Box::new(one_interval_delta),
        Box::new(one_interval_delta_necessary_packets),
        Box::new(conn_interval_sim),
        Box::new(conn_interval_only_gcd_sim),
        Box::new(capture_chance_sim),
    ]; // capture_chance_sim
    run_tasks(tasks, params, bars);

    println!("Connection interval done");
}


fn gcd_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("gcd.png");

    const NUMBER_SIMS : u32 = 1000;



    let pb = bars.lock().unwrap().add(ProgressBar::new(NUMBER_SIMS as u64));
    drop(bars);
    pb.set_style(ProgressStyle::default_bar()
    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}")
    .progress_chars("#>-"));
    let pb = Mutex::new(pb);

    File::create(file_path.clone()).expect("Failed to create plot file");

    //let todo = Mutex::new(NUMBER_SIMS);
    //println!("voor simulatie");
    // Do a simulations
    let sims = (0..NUMBER_SIMS).map(|_| ChaCha20Rng::seed_from_u64(rng.next_u64()))
        .collect_vec();
        
    let sims = sims
        .into_par_iter()
        .filter_map(| mut rng| {
            //println!("Sim {}")
            // own ppm 20
            //const ROUND_THRESS : u64 = (625f64 * 7.5 /  (16f64 + 24f64)*(1000000f64/(500f64+20f64))) as u64;
            //println!("{} round thress", ROUND_THRESS);

            // simulate connection until GCD of sub 1s are found
            // gen random connection
            let mut connection = BleConnection::new(&mut rng, None);
            while connection.connection_interval > ROUND_THRESS as u32 {connection = BleConnection::new(&mut rng, None)}
            let nb_sniffers = rng.gen_range(1u8..=37);
            let packet_loss = rng.gen_range(0.0..0.9);

            // Simulate next channels 
            // Get a random channels sequence to do
            let mut channels = connection.chm.to_vec().into_iter().enumerate().filter(|(_,used)| *used).map(|(c, _)| c as u8).collect_vec();
            channels.shuffle(&mut rng);
            channels.truncate(nb_sniffers as usize);
            


            //println!("Voor take 10");
            // Get 10 sub thress deltas
            let mut drift_since_last = 0;
            let mut events_since_last = 0;
            let mut counter = 0u64;
            let mut prev = (0..).filter_map(|_| 
                if channels.contains(&connection.next_channel())  && rng.gen_range(0.0..1.0) <= 1.0 - packet_loss {
                    Some(connection.cur_time)
                } else {
                    None
                })
            .next().unwrap();
            //println!("Voor deltass, begin take 10");
            let deltas = (0..).filter_map(|_| {
                let ideal = connection.cur_time as i64 + connection.connection_interval as i64;
                let channel = connection.next_channel();
                events_since_last += 1;
                drift_since_last += connection.cur_time as i64 - ideal;
                counter += 1;
                if counter > 1000000000 {
                    panic!("")
                }
                if channels.contains(&channel) && rng.gen_range(0.0..1.0) <= 1.0 - packet_loss {
                    
                    let delta = connection.cur_time - prev;
                    prev = connection.cur_time;
                    let drift = drift_since_last;
                    let events = events_since_last;
                    drift_since_last = 0;
                    events_since_last = 0;
                    if delta < ROUND_THRESS {
                        if drift.abs() > 624 {
                            panic!("Drift was too big yet got through thresshold, {} {} {} {}", events, drift, delta, connection.connection_interval)
                        }
                        let mod_1250 = delta %1250; // round to closest 1250 multiple
                        Some(if mod_1250 > 625 {delta + (1250 - mod_1250)} else {delta - mod_1250})
                    } else {None} } else {None}
            }).take(10).collect_vec();


            //println!("na take 10");
            if let Ok(p) = pb.lock() { p.inc(1);}


            // Check when gcd is conn_interval
            let mut running_gcd = *deltas.first().unwrap(); // max possible


            let gcd_progress =deltas.into_iter().map(|delta| {let b = running_gcd;running_gcd = gcd(running_gcd, delta); if running_gcd < connection.connection_interval as u64 {println!("{} and {} led to to small gcd {} for {}", b, delta, running_gcd, connection.connection_interval)}; running_gcd as u32}).inspect(|g| assert!(*g >= connection.connection_interval)).collect_vec();
            //todo.lock().map(|mut x| *x = x.wrapping_sub(1));
            //println!("todo {}", todo.lock().unwrap());
            gcd_progress.into_iter().enumerate().find_map(|(pos, g)| if g == connection.connection_interval {Some(pos as u32 + 1)} else {None})
        })
        .collect::<Vec<_>>();
        
    //println!("Out of loop");
        
    if let Ok(p) = pb.lock() { p.finish_with_message("Done");}

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption(format!("GCD deltas before found from {} simulations", NUMBER_SIMS), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d((1..10u32).into_segmented(), 0.0..1.02f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Cumulative frequency")
        .x_desc("#Deltas under round threshold")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw observed
    let color = Palette99::pick(0);
    let observed = Histogram::vertical(&events_chart)
            .style(RED.mix(0.5).filled())
            .data(sims.into_iter().flat_map(|x| (x..=10).map(|y|(y, 1.0 / NUMBER_SIMS as f64))));
    
    events_chart.draw_series(observed).unwrap()
        .label("Observed")
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

    // Draw theoretical
    let color = Palette99::pick(1);
    // Multiple has worst case chance of being multiple of 2 as well -> Geometric
    let dist = Geometric::new(1.0/ 2.0).unwrap();
    // x range is nb used channels
    let theoretical = PointSeries::of_element(
        (1..=10_u32).map(|nb_deltas| (SegmentValue::CenterOf(nb_deltas), dist.cdf(nb_deltas as f64 + 0.5))),
        8,
        color.to_rgba().filled(),
        &{
            move |coord, size, style| {
                EmptyElement::at(coord) // have text and circle be relative to this
                + Circle::new((0, 0), size, style)   + Text::new("", (0, 15), ("sans-serif", 8))
            }
        },
    );
    events_chart.draw_series(theoretical).unwrap()
    .label("Theoretical worst case 1/2")
    .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}


fn too_much_drift<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut _rng = params.rng;
    file_path.push("drift_thress_ok_freq.png");

    let nb_sniffers = vec![1u8, 5, 10, 25];

    File::create(file_path.clone()).expect("Failed to create plot file");

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption("Frequency of GCD candidates delta time (37 channels)", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(7500..(ROUND_THRESS as u32), 0.0..1.02f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Frequency")
        .x_desc("Connection interval in ms")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    for (idx, nb_sniffer) in nb_sniffers.into_iter().enumerate()
    {
        let dist = Geometric::new(nb_sniffer as f64 / 37.0).unwrap();
        // Draw theoretical
        let color = Palette99::pick(idx);
        // x range is nb used channels
        let theoretical = LineSeries::new(
            (7500..(ROUND_THRESS as u32)).step_by(1250).map(|conn_int| (conn_int, dist.cdf(0.5 + (ROUND_THRESS/conn_int as u64) as f64))),
            color.to_rgba().filled()
        );
        events_chart.draw_series(theoretical).unwrap()
        .label(format!("{} sniffers", nb_sniffer))
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
    }
    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}

#[allow(dead_code)]
fn mean_drifters<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut _rng = params.rng;
    file_path.push("mean_drifters.png");


    File::create(file_path.clone()).expect("Failed to create plot file");

    let capture_chances = vec![0.95f64, 0.8,0.6, 0.3];

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption("Frequency of GCD candidates delta time (37 channels)", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(plotters::prelude::IntoLinspace::step(7500..4_000_001_u32, 1250), 0.0..1.02f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Frequency")
        .x_desc("Connection interval in ms")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    
    for (idx,capture_chance) in capture_chances.into_iter().enumerate() {
        let dist = Geometric::new(capture_chance).unwrap();
        // Draw theoretical
        let color = Palette99::pick(idx);
        // x range is nb used channels
        let theoretical = LineSeries::new(
            events_chart.x_range().map(|conn_int| {
                //let worst_case_drifted = conn_int as f64 * ( 1.0 / (1.0 + 500.0/1000000.0));
                // Dit bovenste bereken je als je 90% zeker bent dat de kleinste die je zag een deviation is van de echt conn_interval
                // Centrale limietstelling (betrouwbaarheidsinterval) voor uniforme over +- deviation
                // E = worst case drifted en Var is ((lowest*1/(1+500/1000000))^2)/12
                // Dan moet je n bepalen zodat 90% betrouwbaarheid in de 625 band rond midden ligt
                // Dan wacht je tot je n andere hebt die niet meer dan 1.5 keer de eerste zijn
                // Dan neem je de round van het gemiddelde als conn interval
                // PLot n voor con interval en success verdeling over alle conn_interval heen: zou moeten zelfde zijn als je wou door je BI te kiezen = 1 getal
                // TODO die dit voor paar verschillende capture chancen
                // TODO voor een gegeven capture chance is de verwachte waarde # wachten dan
                // TODO P(zal in aanmerking komen)*nodig voor GCD + (1-P(aanmerking))*nodig voor CLT
                let mut _needed_n = 0;
                for n in 1..=8u8 {
                    let max_drift = conn_int as f64 * (500.0/1000000.0);
                    let left = conn_int as f64 - max_drift;
                    let right = conn_int as f64 + max_drift;
                    let left = left.floor() as i16;
                    let right = right.ceil() as i16;
                    // Change drift left and drift right too much
                    let percentage_ok = bates_cdf(624.5, left, right, n) - bates_cdf(-624.5, left, right, n);
                    if percentage_ok > 0.8 {
                        _needed_n = n;
                        break;
                    }
                }

                (conn_int, dist.cdf(0.5 + (ROUND_THRESS/conn_int as u64) as f64))
            }),
            color.to_rgba().filled()
        );
        events_chart.draw_series(theoretical).unwrap()
        .label(format!("P(capture)={:.2}", capture_chance))
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

    }
        

    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}


/// a: left limit
/// b: right limit
/// n: number of samples
/// x en return: P(mean of seen samples < x)
///
///Cumulative Distribution Function (CDF)
///
///               1   i
///  F(x;a,b,n) = --  Σ[(-1)^k·nCk·(r-k)^n]
///               n! k=0
///
///  where
///
///    r = n[(x-a)/(b-a)]
///    i = greatest integer ≤ r
///    nCk = binomial coefficient
///
/// http://www.statext.com/android/batesdist.html
///
#[allow(clippy::many_single_char_names)]
fn bates_cdf(x: f64, a: i16, b: i16, n: u8) -> f64 {
    // Rescale x
    let nx = n as f64 * (x-a as f64)/(b as f64-a as f64);
    let sum_inner = |k: u8| binomial(n, k) as f64 * (nx - k as f64).powi(n as i32);
    let s = (0..=(nx as u8)).map(|k| if k.is_even() {sum_inner(k)} else {- sum_inner(k)}).sum::<f64>();
    // n is small and couldnt find no_std factorial crates
    // 8! = 40320 , will overflow u16 from 9. Either way 9 is infeasable for listening
    //assert!(n <= 8, "bates cdf n must be 8 or lower");
    let fact = (2..=n).fold(1u64, |prev, i| prev * i as u64) as f64;
    s / fact 
}
fn bates_cdf_plot<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut _rng = params.rng;
    file_path.push("bates.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let ns = vec![1u8, 2, 4, 8];

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption("Bates distribution cdf for +/-2000ms drift", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(-2000..2000, 0.0..1.01f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("CDF")
        .x_desc("Uniform interval")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    
    for (idx,n) in ns.into_iter().enumerate() {
        let color = Palette99::pick(idx);
        // x range is nb used channels
        let theoretical = LineSeries::new(
            (-1999..1999).map(|t| {
                (t , bates_cdf(t as f64, -2000, 2000, n))
            }),
            color.to_rgba().filled()
        );
        events_chart.draw_series(theoretical).unwrap()
        .label(format!("n={}", n))
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

    }
    
    events_chart.draw_series(LineSeries::new(
        vec![(0,0.0), (0, 1.0)].into_iter(),
        RED.stroke_width(2)
    )).unwrap()
        .label("Middle")
        .legend(move |(x, y)| Circle::new((x, y), 4, RED.filled()));

    
    events_chart.draw_series(LineSeries::new(
        vec![(-625,0.0), (-625, 1.0)].into_iter(),
        GREEN.stroke_width(2)
    )).unwrap()
        .label("625 bands")
    .legend(move |(x, y)| Circle::new((x, y), 4, GREEN.filled()));
    events_chart.draw_series(LineSeries::new(
        vec![(625,0.0), (625, 1.0)].into_iter(),
        GREEN.stroke_width(2)
    )).unwrap();


    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}


fn bates_necessary_n<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut _rng = params.rng;
    file_path.push("bates_necessary_n.png");


    File::create(file_path.clone()).expect("Failed to create plot file");

    let succes_percentage = vec![0.95f64, 0.9, 0.85, 0.8];

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption("Necessary connection interval to not surpass round threshold", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(plotters::prelude::IntoLinspace::step(7500..4_000_001_u32, 1250), 0..10u32)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Necessary n")
        .x_desc("Connection interval in ms")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    for (idx, success_thress) in succes_percentage.into_iter().enumerate() {
        let color = Palette99::pick(idx);
        // x range is nb used channels
        let theoretical = LineSeries::new(
            events_chart.x_range().filter_map(|conn_int| {
                let max_drift = conn_int as f64 * ((500.0 + 20.0)/1000000.0);
                let left =  (- max_drift).floor() as i16;
                let right =  max_drift.ceil() as i16;
                for n in 1..=10u8 {
                    // Change drift left and drift right too much
                    let percentage_ok = bates_cdf(624.5, left, right, n) - bates_cdf(-624.5, left, right, n);
                    if percentage_ok > success_thress {
                        return Some((conn_int, n as u32))
                    }
                }
                None
            }),
            color.to_rgba().filled()
        );
        events_chart.draw_series(theoretical).unwrap()
        .label(format!("{:.2} success rate", success_thress))
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

    }
        

    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}


fn one_interval_delta<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut _rng = params.rng;
    file_path.push("one_interval_delta.png");


    File::create(file_path.clone()).expect("Failed to create plot file");

    let capture_chances = vec![(10,0.7), (19, 0.7), (5, 0.9), (1, 0.8)];
    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        //.right_y_label_area_size(80)
        .caption("#packets necessary to observe conn_interval = delta", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(plotters::prelude::IntoLinspace::step(0..20u32, 1), 0.0..1.01f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");
        //.set_secondary_coord(0..20u32, 0..300u32);

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Probability captured 2 subsequent packets")
        .x_desc("#Packets")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();
/*
    events_chart.configure_secondary_axes()
        .y_desc("Expected #events necessary for #packets")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();
        */

    for (idx, (nb_sniffs, phy_ch)) in capture_chances.into_iter().enumerate() {
        let capture_chance = nb_sniffs as f64 / 37.0 * phy_ch;
        let dist = Geometric::new(capture_chance).unwrap();
        let _expected_events_for_one_packet = dist.mean().ceil() as u32;
        let color = Palette99::pick(idx);
        // x range is nb used channels
        let theoretical = LineSeries::new(
            (2..20u32).map(|nb_packets| {
                    (nb_packets, 1.0 - dist.cdf(nb_packets as f64 - 1.0 )) // is nb deltas = packets - 1
                }),
            color.to_rgba().filled()
        );
        events_chart.draw_series(theoretical).unwrap()
        .label(format!("{} sniffers, {:.2}  packet loss -> {:.2} P(capture)", nb_sniffs, 1.0 - phy_ch, capture_chance))
        .legend(move |(x, y)| Circle::new((x, y), 4, Palette99::pick(idx).filled()));


        //let theoretical = LineSeries::new(
        //    (1..20u32).map(|nb_packets| {
        //            (nb_packets, expected_events_for_one_packet * nb_packets)
        //        }),
        //    color.to_rgba().filled()
        //);
        //events_chart.draw_secondary_series(theoretical).unwrap();
        //.label(format!("{} sniffers, {:.2}  packet loss", nb_sniffs, 1.0 - phy_ch))
        //.legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

    }
        

    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}

/// p: chance of success
/// wanted_probability: chance of success you want
/// return: necessary repetitions to get the wanted probability
fn geo_qdf(p: f64, wanted_probability : f64) -> u32 {
    let raw = (1.0f64 - wanted_probability).log(1.0 - p);
    //println!("{}", raw);
    raw.ceil() as u32 
}

#[allow(dead_code)]
fn geo_cdf(p: f64, occurences: u32) -> f64 {
    assert!(occurences > 0);
    1.0f64 - (1.0 - p).powi(occurences as i32)
}

#[cfg(test)]
mod geo_test {
    use super::{geo_qdf, geo_cdf};
    use statrs::distribution::Geometric;
    use statrs::distribution::Univariate;
    #[test]
    fn geo() {
        let success_chance = 0.03;
        let target_chance = 0.141266;
        let required_repitions = 5;
        let dist = Geometric::new(success_chance).unwrap();
        assert!((geo_cdf(success_chance, required_repitions) - target_chance).abs() < 0.0001, "{} was not {}", geo_cdf(success_chance, required_repitions), target_chance);
        assert!((dist.cdf(required_repitions as f64) - target_chance).abs() < 0.00001);
        assert_eq!(geo_qdf(success_chance, target_chance - 0.01), required_repitions);
        let calced = geo_qdf(success_chance, target_chance - 0.001);
        assert_eq!(calced, required_repitions);
        let manual = geo_cdf(success_chance, calced);
        let dist_geo_cdf = dist.cdf(calced as f64);
        assert!((dist_geo_cdf - manual).abs() < 0.0001, "{} {} {}", dist_geo_cdf, manual, calced);
        let my_geo_qdf = geo_qdf(success_chance, target_chance - 0.001);
        assert_eq!(my_geo_qdf, required_repitions);
        let my_geo_cdf = geo_cdf(success_chance, calced);
        assert!((my_geo_cdf - target_chance).abs() < 0.00001, "{} was not {}", my_geo_cdf, target_chance);
        assert!((geo_cdf(success_chance, my_geo_qdf) - target_chance).abs() < 0.0001);
        let should_be_x = dist.cdf(calced as f64 + 0.5);
        assert!((should_be_x - target_chance).abs() < 0.01, "{} not {}", should_be_x, target_chance)
    }
}


#[allow(clippy::mut_range_bound)]
fn one_interval_delta_necessary_packets<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut _rng = params.rng;
    file_path.push("one_interval_delta_necessary_packets.png");


    File::create(file_path.clone()).expect("Failed to create plot file");

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption("#events for 90% success rate", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(0.0..1.01f64, 0..1000u32)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("necessary #events")
        .x_desc("capture chance")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();


    let mut last_n = 0;
    let mut points = vec![];
    for cp in (1..100).rev() {
        let capture_chance = cp as f64 * 0.01;
        let necessary_nb_packets = geo_qdf(capture_chance, 0.9);
        // TODO is binomiaal met wat is n op necessary_nb_packets sucesses of meer voor 0.9 zeker -> qdf(0.9) binom (capture_chance)
        for cur_n in last_n..1000 {
            let dist = Binomial::new(capture_chance, cur_n).unwrap();
            if 1.0 -  dist.cdf(necessary_nb_packets as f64 - 0.5) > 0.9 {
                last_n = cur_n;
                points.push((capture_chance, cur_n as u32));
                break
            }
        }
    }
    let mean = LineSeries::new(
        points.into_iter(),
        RED.to_rgba().filled()
    );
    events_chart.draw_series(mean).unwrap()
    .label("upper")
    .legend(move |(x, y)| Circle::new((x, y), 4, RED.filled()));

    let mut last_n = 0;
    let mut points = vec![];
    for cp in (1..100).rev() {
        let capture_chance = cp as f64 * 0.01;
        let necessary_nb_packets = geo_qdf(capture_chance, 0.9);
        // TODO is binomiaal met wat is n op necessary_nb_packets sucesses of meer voor 0.9 zeker -> qdf(0.9) binom (capture_chance)
        for cur_n in last_n..1000 {
            let dist = Binomial::new(capture_chance, cur_n).unwrap();
            if 1.0 -  dist.cdf(necessary_nb_packets as f64 - 0.5) > 0.1 {
                last_n = cur_n;
                points.push((capture_chance, cur_n as u32));
                break
            }
        }
    }
    let mean = LineSeries::new(
        points.into_iter(),
        RED.to_rgba().filled()
    );
    events_chart.draw_series(mean).unwrap()
    .label("lower")
    .legend(move |(x, y)| Circle::new((x, y), 4, RED.filled()));

    let mut last_n = 0;
    let mut points = vec![];
    for cp in (1..100).rev() {
        let capture_chance = cp as f64 * 0.01;
        let necessary_nb_packets = geo_qdf(capture_chance, 0.9);
        // TODO is binomiaal met wat is n op necessary_nb_packets sucesses of meer voor 0.9 zeker -> qdf(0.9) binom (capture_chance)
        for cur_n in last_n..1000 {
            let dist = Binomial::new(capture_chance, cur_n).unwrap();
            if dist.mean() >= necessary_nb_packets as f64 {
                last_n = cur_n;
                points.push((capture_chance, cur_n as u32));
                break
            }
        }
    }
    let mean = LineSeries::new(
        points.into_iter(),
        RED.to_rgba().filled()
    );
    events_chart.draw_series(mean).unwrap()
    .label("mean")
    .legend(move |(x, y)| Circle::new((x, y), 4, RED.filled()));



    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}


fn conn_interval_only_gcd_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("conn_interval_only_gcd_sim.png");

    const NUMBER_SIMS : u32 = 1000;
    const CAPTURE_CHANCE: f64 = 0.02;
    const STEP : usize = 20;

    let gdc_thresses = (1u8..=6).collect_vec();


    File::create(file_path.clone()).expect("Failed to create plot file");

    //let todo = Mutex::new(NUMBER_SIMS);
    //println!("voor simulatie");
    // Do a simulations

    let sims = gdc_thresses.into_iter().map(|i| (i,ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();

    let sims = sims
        .into_par_iter()
        .map(| (gcd_thress, mut rng)| {

        

        let sims = (7500u32..(ROUND_THRESS as u32 - 1250)).step_by(1250 * STEP).map(|i| (i,ChaCha20Rng::seed_from_u64(rng.next_u64())))
            .collect_vec();
            
        let sims = sims
            .into_par_iter()
            .map(| (conn_interval, mut rng)| {
                let res = (0..NUMBER_SIMS).map(|_| {
                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, None);
                    connection.connection_interval = conn_interval;

                    // packet stream
                    let mut packet_stream = (1..).map(|_| {connection.next_channel(); connection.cur_time}).filter(|_| rng.gen_range(0.0..1.0) <= CAPTURE_CHANCE);
                    // Get initial packet
                    let mut prev_packet_time = packet_stream.next().unwrap();
                    let mut durations = vec![];
                    let mut gcd_ok_duration = vec![];
                    loop {
                        let new_packet_time  =  packet_stream.next().unwrap();
                        let new_duration = new_packet_time - prev_packet_time;
                        if new_packet_time <= prev_packet_time {
                            panic!("packet time wrapped")
                        }
                        prev_packet_time = new_packet_time;
                        durations.push(new_duration);
                        let new_duration_max = new_duration as f64 / (1.0 + 500.0/1_000_000.0);
                        let new_max_drift = new_duration_max * (500.0/1_000_000.0);
                        // Check if OK for GCD
                        if new_max_drift < 624.5 {
                            gcd_ok_duration.push(new_duration);
                            // + 1 because in theory you start from say I have x, then n more packets. + 1 is the first duration
                            if gcd_ok_duration.len() >= gcd_thress as usize{
                                let conn_int_gs = gcd_ok_duration.iter().map(|d| {let mod_1250 = *d % 1250;if mod_1250 < 625 {*d - mod_1250} else {*d + 1250 - mod_1250}}).collect_vec();
                                for c in conn_int_gs.iter() {
                                    if *c % connection.connection_interval as u64 != 0 {
                                        panic!("Got illegal in conn_interval")
                                    }
                                }

                                let conn_int =   conn_int_gs.into_iter().reduce(gcd).unwrap() as u32;
                                if conn_int < connection.connection_interval { //|| conn_int != conn_interval {
                                    panic!("GCD candidates had wrong rounding")
                                }
                                return (conn_interval == conn_int, conn_int, connection.cur_time - connection.start_time, true)
                            }
                        }
                    }
                }).collect_vec();
                (conn_interval, res)
            })
            .collect::<Vec<_>>();
        (gcd_thress, sims)
    }).collect::<Vec<_>>();
            

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                            // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption(format!("Onlyg gcd mean time and success for {:.2} capture chance, {} sims per point", CAPTURE_CHANCE, NUMBER_SIMS), ("sans-serif", 20))
        .margin(20)
        //.right_y_label_area_size(80)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(7500..(ROUND_THRESS as u32 + 1), 0.0..1.02f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.")
        .set_secondary_coord(7500..(ROUND_THRESS as u32 + 1), 0..1000u64);

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Success rate")
        .x_desc("Connection interval")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    
    events_chart.configure_secondary_axes()
        .y_desc("Total time in seconds")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();

    for (idx, (gcd_thress, dat)) in sims.into_iter().enumerate() {
        let mut success_rates = vec![];
        let mut times = vec![];
        for (conn_interval, dat) in dat.into_iter() {

            let mut successes = 0 ;
            let mut total_time : u64 = 0;
            let nb_samples = dat.len();
            for (success, _calculated_conn, tot_time, _was_gcd) in dat.iter() {
                if *success {successes += 1}
                total_time += *tot_time as u64;
            }
            let success_rate = successes as f64 / nb_samples as f64;
            let mean_time = total_time as f64 / nb_samples as f64;
            // Put mean time to seconds
            let mean_time = (mean_time / 1_000_000.0).round() as u64;
            if 4000000 - conn_interval < 1250 * STEP as u32 {
                //println!("{} {} {}", success_rate, mean_time, conn_interval);
                //dat.iter().for_each(|d| println!("{:?}", d))
            }
            success_rates.push((conn_interval, success_rate));
            times.push((conn_interval, mean_time));
        }


        let color = Palette99::pick(idx);
        let o = LineSeries::new(
            success_rates.into_iter(),
            color.to_rgba().stroke_width(3));
        events_chart.draw_series(o).unwrap()
        .label(format!("{} durations thresshold ", gcd_thress))
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
        //let color = Palette99::pick(idx);
        //let o = LineSeries::new(
            //times.into_iter(),
            //color.to_rgba().stroke_width(3));
        //events_chart.draw_secondary_series(o).unwrap();

    }


    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}


fn conn_interval_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("conn_interval_sim.png");


    const NUMBER_SIMS : u32 = 10;
    const SUCCESS_RATE: f64 = 0.91;
    const STEP : usize = 20;

    // Plus one because of the start assumption of already having a duration
    // another + 1 because my qdf is not x + 1 but x
    let gcd_thress_nb_durations = geo_qdf(1.0/2.0, SUCCESS_RATE) + 2;
    //println!("{} gcd thres", gcd_thress_nb_durations - 2);

    let capture_chances = vec![0.2f64];//, 0.1f64, 0.02f64];


    File::create(file_path.clone()).expect("Failed to create plot file");

    //let todo = Mutex::new(NUMBER_SIMS);
    //println!("voor simulatie");
    // Do a simulations

    let sims = capture_chances.into_iter().map(|i| (i,ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();

    let sims = sims
        .into_par_iter()
        .map(| (capture_chance, mut rng)| {

        
        let necessary_nb_packets = geo_qdf(capture_chance, SUCCESS_RATE);

        let sims = (7500u32..=4000000).step_by(1250 * STEP).map(|i| (i,ChaCha20Rng::seed_from_u64(rng.next_u64())))
            .collect_vec();
            
        let sims = sims
            .into_par_iter()
            .map(| (conn_interval, mut rng)| {
                let res = (0..NUMBER_SIMS).map(|_| {
                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, None);
                    connection.connection_interval = conn_interval;

                    // Enforce the ppm max
                    if connection.connection_interval as f64 * connection.master_ppm as f64 / 1_000_000.0 > 624.5 {
                        let legal_ppm_max = (625.0 * 1_000_000.0 / connection.connection_interval as f64) as u16;
                        if !(156..500).contains(&legal_ppm_max) {
                            panic!("ubvak")
                        }
                        connection.master_ppm = rng.gen_range(10..=legal_ppm_max);
                    }

                    // packet stream
                    let mut packet_stream = (1..).map(|_| {connection.next_channel(); connection.cur_time}).filter(|_| rng.gen_range(0.0..1.0) <= capture_chance);
                    // Get initial packet
                    let mut prev_packet_time = packet_stream.next().unwrap();
                    let mut durations = vec![];
                    let mut gcd_ok_duration = vec![];
                    loop {
                        let new_packet_time  =  packet_stream.next().unwrap();
                        let new_duration = new_packet_time - prev_packet_time;
                        if new_packet_time <= prev_packet_time {
                            panic!("packet time wrapped")
                        }
                        prev_packet_time = new_packet_time;
                        durations.push(new_duration);
                        let new_duration_max = new_duration as f64 / (1.0 + 500.0/1_000_000.0);
                        let new_max_drift = new_duration_max * (500.0/1_000_000.0);
                        // Check if OK for GCD
                        if new_max_drift < 624.5 {
                            gcd_ok_duration.push(new_duration);
                            // + 1 because in theory you start from say I have x, then n more packets. + 1 is the first duration
                            if gcd_ok_duration.len() >= gcd_thress_nb_durations as usize{
                                let conn_int_gs = gcd_ok_duration.iter().map(|d| {let mod_1250 = *d % 1250;if mod_1250 < 625 {*d - mod_1250} else {*d + 1250 - mod_1250}}).collect_vec();
                                for c in conn_int_gs.iter() {
                                    if *c % connection.connection_interval as u64 != 0 {
                                        panic!("Got illegal in conn_interval")
                                    }
                                }

                                let conn_int =   conn_int_gs.into_iter().reduce(gcd).unwrap() as u32;
                                if conn_int < connection.connection_interval { //|| conn_int != conn_interval {
                                    panic!("GCD candidates had wrong rounding")
                                }
                                if conn_interval == conn_int && conn_interval == 57500 {
                                    println!("let conn_interval : u32 = {};\nlet capture_chance : f32 = {};\nlet nb_packets_first_single_interval : u32 = {} + 1;\nlet nb_durations_gcd_thres : u32 = {};\nlet durations : Vec<u32> = vec!{:?};\n\n", conn_interval, capture_chance, necessary_nb_packets, gcd_thress_nb_durations, durations);
                                }
                                return (conn_interval == conn_int, conn_int, connection.cur_time - connection.start_time, true)
                            }
                        }
                        // Check if we reached thresshold
                        if durations.len() as u32 >= necessary_nb_packets {
                            // Find smallest and round
                            let smallest = *durations.iter().min().unwrap();
                            let mod_1250 = smallest % 1250;
                            let conn_int = if mod_1250 < 625 {smallest - mod_1250} else {smallest + 1250 - mod_1250} as u32;
                            // Take it as the conn_interval
                            if conn_interval == conn_int && conn_interval == 2507500 {
                                println!("let conn_interval : u32 = {};\nlet capture_chance : f32 = {};\nlet nb_packets_first_single_interval : u32 = {}+1;\nlet nb_durations_gcd_thres : u32 = {};\nlet durations : Vec<u32> = vec!{:?};\n\n", conn_interval, capture_chance, necessary_nb_packets, gcd_thress_nb_durations, durations);
                            }
                            return (conn_interval == conn_int, conn_int, connection.cur_time - connection.start_time, false)
                        }
                    }
                }).collect_vec();
                (conn_interval, res)
            })
            .collect::<Vec<_>>();
        (capture_chance, sims)
    }).collect::<Vec<_>>();
            

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                            // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 100)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption(format!("Mean time and success for {:.2} minimum success, {} sims per point", SUCCESS_RATE, NUMBER_SIMS), ("sans-serif", 20))
        .margin(20)
        .right_y_label_area_size(80)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(7500..4_000_001_u32, 0.0..1.02f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.")
        .set_secondary_coord(7500..4_000_001_u32, 0..1000u64);

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Success rate")
        .x_desc("Connection interval")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    
    events_chart.configure_secondary_axes()
        .y_desc("Total time in seconds")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();

    for (idx, (capture_chance, dat)) in sims.into_iter().enumerate() {
        let mut success_rates = vec![];
        let mut times = vec![];
        for (conn_interval, dat) in dat.into_iter() {

            let mut successes = 0 ;
            let mut total_time : u64 = 0;
            let nb_samples = dat.len();
            for (success, _calculated_conn, tot_time, _was_gcd) in dat.iter() {
                if *success {successes += 1}
                total_time += *tot_time as u64;
            }
            let success_rate = successes as f64 / nb_samples as f64;
            let mean_time = total_time as f64 / nb_samples as f64;
            // Put mean time to seconds
            let mean_time = (mean_time / 1_000_000.0).round() as u64;
            if 4000000 - conn_interval < 1250 * STEP as u32 {
                //println!("{} {} {}", success_rate, mean_time, conn_interval);
                //dat.iter().for_each(|d| println!("{:?}", d))
            }
            success_rates.push((conn_interval, success_rate));
            times.push((conn_interval, mean_time));
        }


        let color = Palette99::pick(idx);
        let o = LineSeries::new(
            success_rates.into_iter(),
            color.to_rgba().stroke_width(3));
        events_chart.draw_series(o).unwrap()
        .label(format!("{} capture chance", capture_chance))
        .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
        let color = Palette99::pick(idx);
        let o = LineSeries::new(
            times.into_iter(),
            color.to_rgba().stroke_width(3));
        events_chart.draw_secondary_series(o).unwrap();

    }


    // Draws the legend
    events_chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}



struct CaptureParams {
    physical_chance: f64,
    nb_sniffers: u8,
    anchor_point_percentage: f64,
    master_phy: BlePhy,
    slave_phy: BlePhy,
    silence_percentage: f64
}


fn capture_chance_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    // Dir for per nb unused observed
    file_path.push("capture_chance_sim");
    create_dir_all(file_path.clone()).expect("Failed to create plot directory");

    // First one gets taken as the number of events to plot them all together
    let capture_params = vec![CaptureParams {
        physical_chance: 0.8,
        nb_sniffers: 5,
        anchor_point_percentage: 0.9,
        master_phy: BlePhy::Uncoded1M,
        slave_phy: BlePhy::CodedS2,
        silence_percentage: 0.05
    },
    CaptureParams {
        physical_chance: 0.8,
        nb_sniffers: 1,
        anchor_point_percentage: 0.9,
        master_phy: BlePhy::Uncoded1M,
        slave_phy: BlePhy::CodedS2,
        silence_percentage: 0.2
    },
    CaptureParams {
        physical_chance: 0.6,
        nb_sniffers: 10,
        anchor_point_percentage: 0.95,
        master_phy: BlePhy::CodedS8,
        slave_phy: BlePhy::CodedS8,
        silence_percentage: 0.05
    },];
    
    let sims = capture_params.into_iter().map(|n| (n, ChaCha20Rng::seed_from_u64(rng.next_u64())))
    .collect_vec();
    sims.into_par_iter().enumerate().for_each(|(enu,(capture_params, mut urng))| {
        // TODO create dir for capture params

        
        let mut file_path = file_path.clone();
        file_path.push(format!("phy_{:.2}_snifs_{}_ap_{:.2}_{}",capture_params.physical_chance, capture_params.nb_sniffers,capture_params.anchor_point_percentage, enu));
        create_dir_all(file_path.clone()).expect("Failed to create plot directory");


        let sims = (2u8..=37).map(|n| (n, ChaCha20Rng::seed_from_u64(urng.next_u64())))
        .collect_vec();

        const NUMBER_SIMS : u32 = 400;


        // Calculate wait time for 1 exchange
        let one_exchange_time = phy_to_max_time(&capture_params.master_phy) + 152 + 24 + phy_to_max_time(&capture_params.slave_phy) + 152 + 24;
        // If you want to be 90% sure a previous one would have occurred before if there were some, do this
        let necessary_exchanges = geo_qdf(capture_params.physical_chance, capture_params.anchor_point_percentage);
        // TODO breakpoint en check die fucking qdf
        let silence_time = one_exchange_time * necessary_exchanges as u64;
        let time_to_switch = (silence_time as f64 / capture_params.silence_percentage) as u64; // 0.05 = 0.95 percent of time listening

        sims.into_par_iter().for_each(|(nb_used, mut rng)| {
            // PLOT PER NUMBER USED
            const STEP : u32 = 37; // gewoon priem
            let conn_int_sims = (7500u32..=4000000).step_by((1250 * STEP) as usize).filter_map(|conn_interval| {
                if one_exchange_time >= conn_interval as u64 {return None} 
                let sims = (0..NUMBER_SIMS).map(|_|{


                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, Some(nb_used as u8));
                    connection.connection_interval = conn_interval;

                    let mut sniffers_channels_start_time = connection.start_time;
                    let mut sniffers_location = (0..capture_params.nb_sniffers).collect_vec();


                    // Listen for 1000 connection events
                    let captured = (0..1000).map(|_| 
                        {
                            // check if the first packet would have been caught (don't care here about anchor point error)
                            let mut captured = false;
                            let delta_since_start = connection.cur_time - sniffers_channels_start_time;
                            if delta_since_start > silence_time && 
                                sniffers_location.contains(&connection.cur_channel) &&
                                rng.gen_range(0.0..1.0) <= capture_params.physical_chance {
                                    captured = true;
                            }

                            // only at end
                            connection.next_channel();
                            if connection.cur_time >= sniffers_channels_start_time + time_to_switch {
                                sniffers_channels_start_time += time_to_switch;
                                //let mut st  = String::new();
                                sniffers_location.iter_mut().for_each(|s| *s = (*s + capture_params.nb_sniffers) % 37);
                                //.inspect(|s| st.push_str(format!("{} ", **s).as_str()))
                                //st.push('\n');
                                //sniffers_location.iter().for_each(|s| st.push_str(format!("{} ", *s).as_str()));
                                //println!("{}\n\n\n", st);
                                //std::io::stdout().flush();
                            }
                            captured
                        }
                    ).collect_vec();

                    let l =captured.len();
                    captured.into_iter().filter(|b|*b).count() as f64 / l as f64
                }).collect_vec();
                Some((conn_interval, sims))
            }).collect_vec();
                
            // TODO PLOT
            let mut file_path = file_path.clone();
            file_path.push(format!("{}_used.png", nb_used));
            File::create(file_path.clone()).expect("Failed to create plot file");

            const HEIGHT: u32 = 1080;
            const WIDTH: u32 = 1080; // was 1920
                                    // Get the brute pixel backend canvas
            let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
            root_area.fill(&WHITE).unwrap();


            let mut events_chart = ChartBuilder::on(&root_area)
                .set_label_area_size(LabelAreaPosition::Left, 120)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .caption(format!("Capture chance: {} used, {:.2} packet loss, {} sniffers, {:.2} ap, {:.2} silence, {} {} phys", 
                    nb_used, 1.0 - capture_params.physical_chance, capture_params.nb_sniffers, capture_params.anchor_point_percentage, capture_params.silence_percentage
                    , phy_to_string_short(&capture_params.master_phy), phy_to_string_short(&capture_params.slave_phy)), 
                    ("sans-serif", 20))
                .margin(20)
                .build_cartesian_2d(plotters::prelude::IntoLinspace::step(7500u32..4000001, 1250 * STEP).into_segmented(), 0.0..1.05f32)
                .expect("Chart building failed.");
            events_chart
                .configure_mesh()
                .disable_x_mesh()
                .y_desc("Capture chance")
                .x_desc("Connection interval")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw()
                .unwrap();
            
            let boxes = conn_int_sims.iter().map(|(c, s)| Boxplot::new_vertical(SegmentValue::CenterOf(*c), &Quartiles::new(s)));
            
            //let o = LineSeries::new(
            //    probs.iter().map(|p| (p.0 as u32, p.4)),
            //    BLUE.stroke_width(3));
            events_chart.draw_series(boxes).unwrap()
            .label("Observed relative frequency")
            .legend(move |(x, y)| Circle::new((x, y), 4, BLUE.filled()));
            // Draw theoretical
            let theo : f64 =  (1.0 - capture_params.silence_percentage) * capture_params.physical_chance * (capture_params.nb_sniffers as f64 / 37.0) ;
            let c = LineSeries::new(
            vec![(SegmentValue::Exact(7500), theo as f32), (SegmentValue::Exact(4000000),theo as f32)].into_iter(), RED.stroke_width(3));
            events_chart.draw_series(c).unwrap()
            .label("theoretical")
            .legend(move |(x, y)| Circle::new((x, y), 4, RED.filled()));
            // Draws the legend
            events_chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .label_font(("sans-serif", 15))
                .position(SeriesLabelPosition::UpperRight)
                .draw()
                .unwrap();              

        });
    });
}

fn phy_to_max_time(phy: &BlePhy) -> u64 {
    static UNCODED_1M_SEND_TIME: u64 = 2128;
    static UNCODED_2M_SEND_TIME: u64 = 2128 / 2 + 4;
    static CODED_S2_SEND_TIME: u64 = 4542; // AA, CI, TERM1 in S8
    static CODED_S8_SEND_TIME: u64 = 17040;
    match phy {
        BlePhy::Uncoded1M => {UNCODED_1M_SEND_TIME}
        BlePhy::Uncoded2M => {UNCODED_2M_SEND_TIME}
        BlePhy::CodedS2 => {CODED_S2_SEND_TIME}
        BlePhy::CodedS8 => {CODED_S8_SEND_TIME}
    }
}

fn phy_to_string_short(phy: &BlePhy) -> &str {
    match phy {
        BlePhy::Uncoded1M => {"1M"}
        BlePhy::Uncoded2M => {"2M"}
        BlePhy::CodedS2 => {"S2"}
        BlePhy::CodedS8 => {"S8"}
    }
}