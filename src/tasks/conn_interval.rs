use itertools::Itertools;
use plotters::prelude::*;
use rand::{Rng, seq::SliceRandom};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{fs::{create_dir_all, File}, iter::once, mem::needs_drop, ops::RangeBounds};

use statrs::distribution::{Binomial, Erlang, Geometric, Hypergeometric};
use statrs::{distribution::Univariate, statistics::Mean};
use std::f64;
use num::{Integer, integer::{binomial, gcd}};

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
        //Box::new(time_boxplots_per_nb_used),
    ]; // conn_interval_sim
    run_tasks(tasks, params, bars);
}


fn gcd_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("gcd.png");

    const NUMBER_SIMS : u32 = 10000;



    let pb = bars.lock().unwrap().add(ProgressBar::new(NUMBER_SIMS as u64));
    drop(bars);
    pb.set_style(ProgressStyle::default_bar()
    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}")
    .progress_chars("#>-"));
    let pb = Mutex::new(pb);

    File::create(file_path.clone()).expect("Failed to create plot file");

    let todo = Mutex::new(NUMBER_SIMS);
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
                let mut needed_n = 0;
                for n in 1..=8u8 {
                    let max_drift = conn_int as f64 * (500.0/1000000.0);
                    let left = conn_int as f64 - max_drift;
                    let right = conn_int as f64 + max_drift;
                    let left = left.floor() as i16;
                    let right = right.ceil() as i16;
                    // Change drift left and drift right too much
                    let percentage_ok = bates_cdf(624.5, left, right, n) - bates_cdf(-624.5, left, right, n);
                    if percentage_ok > 0.8 {
                        needed_n = n;
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
        let expected_events_for_one_packet = dist.mean().ceil() as u32;
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

fn geo_cdf(p: f64, occurences: u32) -> f64 {
    assert!(occurences > 0);
    1.0f64 - (1.0 - p).powi(occurences as i32)
}

#[cfg(test)]
mod GeoTest {
    use super::{geo_qdf, geo_cdf};
    use statrs::distribution::Geometric;
    use statrs::distribution::Univariate;
    #[test]
    fn geo() {
        let p = 0.03;
        let x = 0.141266;
        let y = 5;
        let dist = Geometric::new(p).unwrap();
        assert!((geo_cdf(p, y) - x).abs() < 0.0001, "{} was not {}", geo_cdf(p, y), x);
        assert!((dist.cdf(y as f64) - x).abs() < 0.00001);
        assert_eq!(geo_qdf(p, x - 0.01), y);
        let calced = geo_qdf(p, x - 0.001);
        assert_eq!(calced, y);
        let manual = geo_cdf(p, calced);
        let g = dist.cdf(calced as f64);
        assert!((g - manual).abs() < 0.0001, "{} {} {}", g, manual, calced);
        let g = geo_qdf(p, x - 0.001);
        assert_eq!(g, y);
        let q = geo_cdf(p, calced);
        assert!((q - x).abs() < 0.00001, "{} was not {}", q, x);
        assert!((geo_cdf(p, g) - x).abs() < 0.0001);
        let should_be_x = dist.cdf(calced as f64 + 0.5);
        assert!((should_be_x - x).abs() < 0.01, "{} not {}", should_be_x, x)
    }
}

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


fn conn_interval_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("conn_interval_sim.png");

    const GDC_THRESS : usize = 2;
    const NUMBER_SIMS : u32 = 100;
    const SUCCESS_RATE: f64 = 0.9;

    let capture_chances = vec![0.2f64, 0.1f64, 0.02f64];


    File::create(file_path.clone()).expect("Failed to create plot file");

    //let todo = Mutex::new(NUMBER_SIMS);
    //println!("voor simulatie");
    // Do a simulations

    let sims = capture_chances.into_iter().map(|i| (i,ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();

    let sims = sims
        .into_par_iter()
        .map(| (capture_chance, mut rng)| {

        
        let necessary_nb_packets = geo_qdf(capture_chance, SUCCESS_RATE);

        let sims = (7500u32..=4000000).map(|i| (i,ChaCha20Rng::seed_from_u64(rng.next_u64())))
            .collect_vec();
            
        let sims = sims
            .into_par_iter()
            .map(| (conn_interval, mut rng)| {
                let res = (0..NUMBER_SIMS).map(|_| {
                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, None);
                    connection.connection_interval = conn_interval;

                    // packet stream
                    let mut packet_stream = (1..).map(|_| {connection.next_channel(); connection.cur_time}).filter(|_| rng.gen_range(0.0..1.0) <= capture_chance);
                    // Get initial packet
                    let mut prev_packet_time = packet_stream.next().unwrap();
                    let mut durations = vec![];
                    let mut gcd_ok_duration = vec![];
                    loop {
                        let new_packet_time  =  packet_stream.next().unwrap();
                        let new_duration = prev_packet_time - new_packet_time;
                        prev_packet_time = new_packet_time;
                        durations.push(new_duration);
                        // Check if OK for GCD
                        if new_duration as f64 * (500.0/1_000_000.0) < 625.0 {
                            gcd_ok_duration.push(new_duration);
                            if gcd_ok_duration.len() >= GDC_THRESS {
                                let conn_int = gcd_ok_duration.into_iter().map(|d| if d % 1250 < 625 {d - d % 1250} else {d + 1250 - d % 1250})
                                    .reduce(gcd).unwrap() as u32;
                                return (conn_interval == conn_int, conn_int, connection.cur_time, true)
                            }
                        }
                        // Check if we reached thresshold
                        if durations.len() as u32 >= necessary_nb_packets {
                            let smallest = *durations.iter().min().unwrap();
                            let smallest_max = smallest + (smallest as f64 * (500.0/1_000_000.0)).ceil() as u64;
                            let smallest_min = smallest - (smallest as f64 * (500.0/1_000_000.0)).ceil() as u64;
                            let multiple = if  let Some(mutliple) = durations.iter()
                                .filter(|d| **d - (**d as f64 * (500.0/1_000_000.0)).ceil() as u64 > smallest_max).min() { *mutliple} 
                                else {
                                    fn rec_pos(durations: &mut Vec<u64>, mut running_duration: u64) -> Vec<u64> {
                                        if durations.len() == 1 {
                                            return vec![]
                                        }
                                        
                                        let new = durations.pop().unwrap();
                                        running_duration += new;
                                        let mut rec = rec_pos(durations, running_duration);
                                        rec.push(running_duration);
                                        rec
                                    }

                                    durations.reverse();
                                    let r = rec_pos(&mut durations, 0);
                                    if let Some(mul) = r.into_iter().filter(|d| *d - (*d as f64 * (500.0/1_000_000.0)).ceil() as u64 > smallest_max).min() {
                                        mul
                                    }
                                    else {
                                        continue;
                                    }
                                };
                            let multiple_max = multiple + (multiple as f64 * (500.0/1_000_000.0)).ceil() as u64;
                            let multiple_max_in_1250 = ((multiple_max - multiple_max % 1250) / 1250) as u32; // Always round down, round up would not be possible
                            let smallest_max_in_1250 = ((smallest_max - smallest_max % 1250) / 1250) as u32; 
                            let multiple_min = multiple - (multiple as f64 * (500.0/1_000_000.0)).ceil() as u64;
                            let multiple_min_in_1250 = ((multiple_min + 1250 - multiple_min % 1250) / 1250) as u32; // Always round down, round up would not be possible
                            let smallest_min_in_1250 = ((smallest_min + 1250 - smallest_min % 1250) / 1250) as u32;
                            let conn_int = (smallest_min_in_1250..=smallest_max_in_1250).cartesian_product(multiple_min_in_1250..=multiple_max_in_1250).find_map(|(smallest, multiple)| if multiple % smallest == 0 {Some(smallest * 1250)} else {None}).expect("Smallest was not factor of multiple!");
                            return (conn_interval == conn_int, conn_int, connection.cur_time, false)
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
        .caption(format!("Mean time and success, {} sims per point", NUMBER_SIMS), ("sans-serif", 20))
        .margin(20)
        .right_y_label_area_size(80)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(7500..4_000_001_u32, 0.0..1.02f64)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.")
        .set_secondary_coord(7500..4_000_001_u32, 0..1_000_000u64);

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
            for (success, _calculated_conn, tot_time, _was_gcd) in dat {
                if success {successes += 1}
                total_time += tot_time as u64;
            }
            let success_rate = successes as f64 / nb_samples as f64;
            let mean_time = total_time as f64 / nb_samples as f64;
            // Put mean time to seconds
            let mean_time = (mean_time / 1_000_000.0).round() as u64;
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



fn time_boxplots_per_nb_used<R: RngCore + Send + Sync>(params: SimulationParameters<R>, bars: Arc<Mutex<MultiProgress>>)  {

    const NUMBER_SIMS : u32 = 100;

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("time_boxplots_per_nb_used");
    create_dir_all(file_path.clone()).expect("Failed to create plot directory");



    let nb_sniffers = vec![1u8, 5, 10, 15, 25, 37];
    let max_error_rates = vec![0.1f64];
    let packet_loss = vec![0.1f64];

    let nt = NUMBER_SIMS as usize * 36 * nb_sniffers.len() * max_error_rates.len() * packet_loss.len();

    // make new progress bar
    let pb = bars.lock().unwrap().add(ProgressBar::new(nt as u64));
    pb.set_style(ProgressStyle::default_bar()
    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}")
    .progress_chars("#>-"));
    let pb = Mutex::new(pb);

    let todo = Mutex::new((2u8..38).collect_vec());

    let plots = nb_sniffers.into_iter().cartesian_product(max_error_rates.into_iter().cartesian_product(packet_loss.into_iter()))
    .map(|(b, (e, p))| (b, e, p,ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
    plots.into_par_iter().for_each(|(nb_sniffers, max_error, packet_loss,mut rng)| {

        // simulate for every possible real used
        let sims = (2u8..=37).map(|i| (i, ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
        // Contains (actual_nb_used, vec over simulation with ((success, not no_solution), nb_bfs, extra_packets after channel map))
        /*
        let sims = sims.into_par_iter().map(|(nb_used, mut rng)|{
            let dist =  Geometric::new((1.0-packet_loss)*(1.0/nb_used as f64)).unwrap();
            let capture_chance = dist.cdf(events as f64 + 0.5);


            let sims = (0..NUMBER_SIMS).map(|i| (i, ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
            let resses = sims.into_par_iter().map(|(_ns, mut rng)| {
                // gen random connection
                let mut connection = BleConnection::new(&mut rng, Some(nb_used as u8));
                // Get a random channels sequence to do
                let mut channels = connection.chm.to_vec().into_iter().enumerate().map(|(c, used)| Occ{ channel: c as u8, used}).collect_vec(); // (channel, is_used)
                channels.shuffle(&mut rng);

                // Simulate the channel map stage                        
                let mut observerd_packets = channels.into_iter().filter_map(|occ| 
                    if occ.used {(0..events).filter_map(|_|
                        // If any is captured and the channel is the channel it is not unused -> negate this -> !(..)any
                        if connection.next_channel() == occ.channel && rng.gen_range(0.0..1.0) <= capture_chance {
                            Some((connection.cur_event_counter, connection.cur_channel))
                        } else {None}).next()} // short circuit first channel occurrence
                    else {(0..events).for_each(|_| {connection.next_channel();}); None} // let connection jump events_to_wait
                    ).collect_vec();

                // Put them from the relative offset
                let relative_event_offset = observerd_packets.first().unwrap().0;
                observerd_packets.iter_mut().for_each(|p| p.0 = (p.0).wrapping_sub(relative_event_offset));
                // Get channel map
                let chm = observerd_packets.iter().fold(0u64, |chm, (_, channel)| chm | (1 << *channel));
                let (channel_map_bool_array,_, _, nb_used_observed) =  generate_channel_map_arrays(chm);
                let observed_used = (0u8..37).filter(|c| channel_map_bool_array[*c as usize]).collect_vec();

                // brute force and wait for new packets as long as you have no single solution
                let mut extra_packets = 0u32;
                let mut result = brute_force(extra_packets, bfs_max, nb_used, connection.channel_map, relative_event_offset,observerd_packets.as_slice(), chm, thress, events, packet_loss, connection.channel_id);
                while let CounterInterval::MultipleSolutions = &result.0 {
                    // Get extra packet
                    extra_packets += 10;
                    // Get random next channel to listen for extra packet
                    let channel = observed_used[rng.gen_range(0..observed_used.len())];
                    // Listen until you hear one
                    let mut next_one = (0..).map(|_| {connection.next_channel(); (connection.cur_event_counter.wrapping_sub(relative_event_offset), connection.cur_channel)}).filter(|c| c.1 == channel  && rng.gen_range(0.0..1.0) <= capture_chance).take(10).collect_vec();
                    // Add to observed
                    observerd_packets.append(&mut next_one);

                    //if nb_used == 37 && extra_packets > 50 {println!("Large packets {} for {}", extra_packets, nb_used)};

                    // Brute force again
                    result = brute_force(extra_packets, bfs_max, nb_used, connection.channel_map, relative_event_offset, observerd_packets.as_slice(), chm, thress, events, packet_loss, connection.channel_id);

                }

                //if nb_used == 37 { println!("Completed sim {} of {} for {}: {:?} {}", ns, NUMBER_SIMS, nb_used, &result, extra_packets)};
                //if let Ok(p) = pb.lock() { p.inc(1); p.set_message(format!("{}", nb_used)) }
                if let CounterInterval::ExactlyOneSolution(first_packet_actual_counter, found_chm) = &result.0 {
                    (((*first_packet_actual_counter == relative_event_offset && connection.channel_map == *found_chm), true), result.1, extra_packets, nb_used - nb_used_observed)
                }
                else {
                    ((false,false), result.1, extra_packets, nb_used - nb_used_observed)
                }
            }).collect::<Vec<_>>();
            todo.lock().unwrap().retain(|x| *x != nb_used);
            println!("{:?} todo", todo.lock().unwrap());
            (nb_used, resses)
        }).collect::<Vec<_>>();
        */
        //let sims = Vec::new();

        let mut this_path = file_path.clone();
        this_path.push(format!("time_{:02}-sniffers_{:.2}-err_{:.2}-pl.png",  nb_sniffers, max_error, packet_loss));
        File::create(this_path.clone()).expect("Failed to create plot file");


        // return 3 iterators with the boxlots
        /*
        let (successes, fns): (Vec<_>, Vec<_>) = sims.into_iter().map(|(nb_used : u8, data)| {


            #[allow(clippy::type_complexity)]
            //let (successes,(no_sols, (bfs, (extra, fns)))): (Vec<_>, (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>)))) = data.into_par_iter().inspect(|d| println!("{:?}", d)).map(|d| (d.0.0,(!d.0.1, (d.1, (d.2, d.3))))).unzip();

            let successes = successes.into_iter().filter(|b| *b).count() as f64 / NUMBER_SIMS as f64;
            let fns_quarts = Quartiles::new(&fns);

            ( (nb_used, successes), Boxplot::new_vertical(nb_used as u32, &fns_quarts
            ))
        }).unzip();
        */
        let fns_quarts = Quartiles::new(&[0u32]);
        let dummy = Boxplot::new_vertical(0, &fns_quarts);

        let successes : Vec<(u8, f64)> = Vec::new();
        let fns = vec![dummy];

        const HEIGHT: u32 = 1080;
        const WIDTH: u32 = 1080; // was 1920

        // Successes/errors
        let root_area = BitMapBackend::new(this_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();
        let mut events_chart = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 120)
            .set_label_area_size(LabelAreaPosition::Bottom, 60)
            .caption(format!("time: {:02} sniffers, {:.2} error, {:.2} packet loss",  nb_sniffers, max_error, packet_loss), ("sans-serif", 20))
            .margin(20)
            .right_y_label_area_size(80)
            .build_cartesian_2d(1..38u32, 0.0..1.05f32)
            .expect("Chart building failed.")
            .set_secondary_coord(1..38u32, 0.0..10.05f32); 
        events_chart
            .configure_mesh()
            .disable_x_mesh()
            .y_desc("Error rate")
            .x_desc("Actual used channels")
            .label_style(("sans-serif", 20)) // The style of the numbers on an axis
            .axis_desc_style(("sans-serif", 20))
            .draw()
            .unwrap();
        events_chart.configure_secondary_axes()
            .y_desc("False negatives")
            .label_style(("sans-serif", 20)) // The style of the numbers on an axis
            .axis_desc_style(("sans-serif", 20))
            .draw().unwrap();
        let o = LineSeries::new(
            successes.into_iter().map(|(nb_used, succ)| (nb_used as u32, 1.0 - succ as f32)),
            RED.to_rgba().stroke_width(3));
        events_chart.draw_series(o).unwrap();
        events_chart.draw_secondary_series(fns).unwrap();

    });
}
