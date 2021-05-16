use itertools::Itertools;
use jambler::ble_algorithms::{
    access_address::is_valid_aa,
    csa2::{calculate_channel_identifier, csa2_no_subevent, generate_channel_map_arrays},
};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{ fs::{create_dir_all, File}};

use statrs::distribution::{Binomial, Discrete, Geometric};
use statrs::function::factorial::binomial;
use statrs::statistics::OrderStatistics;
use statrs::{distribution::Univariate, statistics::Mean};
use std::f64;

use crate::{SimulationParameters, Task, run_tasks, tasks::BleConnection};

pub fn channel_recovery<R: RngCore + Send + Sync>(mut params: SimulationParameters<R>) {
    params.output_dir.push("channel_recovery");
    create_dir_all(&params.output_dir).unwrap();
    let tasks: Vec<Box<dyn Task>> = vec![
        Box::new(false_negatives),
        Box::new(misclass_chance_vs_combinations),
        Box::new(nb_brute_force),
        Box::new(computing_vs_events),
        Box::new(how_many_to_check),
        Box::new(please),
    ]; // how_many_to_check
    run_tasks(tasks, params);
}

fn please<R: RngCore + Send + Sync>(_params: SimulationParameters<R>) {
    let events = vec![100_u8, 200, 70];
    let nb_unused_observed = vec![2_u8,8,25];
    events.into_iter().for_each(|nb_evenst| {
        nb_unused_observed.iter().for_each(|nb_unused_seen| {
            let probs = (0..=*nb_unused_seen).map(|nb_false_negs| {
                let nb_used = 37 - *nb_unused_seen + nb_false_negs;
                // So now I waited everywhere for nb_evenst events and found nb_unused_seen unused channels (with possible false negatives)
                // Assume nb_false_negs false negatives and thus nb_used actual channels under that assumptions.
                // What is the chance of this occurring?
                let dist = Geometric::new(1.0 / nb_used as f64).unwrap(); // pdf(e): chance of first occurring at e
                let chance_of_hearing_channel = dist.cdf(nb_evenst as f64); // cdf(e): chance of first occurring on or before e
                let all_dist = Binomial::new(chance_of_hearing_channel, nb_used as u64).unwrap(); // pdf(e) = chance of hearing exactly e of nb_used channels
                let prob = all_dist.pmf((nb_used - nb_false_negs) as u64); // the chance of seeing exactly that many true positives (and thus false negatives)
                // The chance of the true positives (no false positives possible)
                //let mut prob = dist.cdf(nb_evenst as f64).powi(nb_used.into()); // cdf(e): chance of first occurring on or before e
                // The chance of the number of false negatives
                //prob *= (1.0 - dist.cdf(nb_evenst as f64)).powi(nb_false_negs.into()); // 1.0 - cdf(e): chance of not occurring on or before e
                // The true negatives have a probability of 1 for not occurring and a probability of 0 for occurring
                let nb_bf = binomial(*nb_unused_seen as u64, nb_false_negs as u64) as u64;
                (nb_false_negs, nb_used, chance_of_hearing_channel, prob, nb_bf)
            }).collect_vec();
            println!("Observed {} unused when waiting for {} events:\n", nb_unused_seen, nb_evenst);
            let s = "| #FN | #used | P(capture) | P(#FN | #used) | #Combinations";
            println!("{}",s);
            s.chars().for_each(|_| print!("-"));
            println!();
            probs.iter().for_each(|p| println!("| {:3} | {:5} | {:9.2}% | {:13.2}% | {}", p.0, p.1, p.2 * 100.0, p.3 * 100.0, p.4));
            //println!("Sum: {}", probs.into_iter().map(|p| p.3).sum::<f64>());
            println!("\n\n\n");

            // TODO does this have to sum to 1? 
            // I think not because for s used channels the number of false negatives probabilities will sum to 1.
            // The number of used channels is a given which I do not know.
            // What I am saying here is that, say 30 used channels, thus with a known false negatives distribution.
            // By brute for all with a probability higher then 10%, when 30 comes up as a candidate,
            // I will find the correct channel map when his chance on the number of false negatives for that amount of seen unused channels,
            // is larger than 10%.
            // So the chance of finding the correct channel map is the sum of the chances of all number of false negatives for 30 for which his chance 
            // for that occurring is above 10%.
            // So not knowing the actual used channels (30 here), I do not have all information for saying what my chances are.
            // So say you brute force all above 10%, then your worst case chance is the one for the actual amount of used channels
            // which has the lowest sum of "bars" (#FNs) for which the bar is above 10%.

            // TODO morgen: JE BENT ER BIJNA!!!!!
            // TODO Doe bovenstaande als simulatie: voor de nb_used = 37 - *nb_unused_seen + nb_false_negs
            // TODO Simuleer 1000 keer en kijk hoeveel keer (relatief)  hij nb_false_negs als false negatives had (used als unused).
            // TODO Check of dit dan overeenstemt
            // TODO Plot dan voor een gegeven aantal unused en events het equivalent van je tabel: 2-y axissen met op ene P(#FN | #used) en andere #Combinations
            // TODO Plot dan (mss eerst tabel) voor een gegeven selectie percentage (10% van hierboven), de som van de kansen voor #FNs die daarboven liggen voor ieder aantal used channels.
            // TODO Selecteer daar een verstandig minimum uit (je worst case slaag percentage)
            // TODO Plot/tabel dan voor gegeven aantal events, hoeveel burte forces (som #Combos boven 10%) je moet doen voor iedere mogelijk aantal geziene unused channels
            // TODO Hopelijk kan je dan een minimum slaag percentage selecteren waarvoor de maximum brute force over alle unused channels computationally feasable is.
            // TODO Herdoe/ pas aan met fysiek capture percentage.
            // TODO Om het aft te maken, plot dan ook je minimum slaag percentage op y as naar het maximum brute force (over gezien unused) (slaag percentage is NIET thresshold percentage).
            // TODO Dan at runtime: gegeven aantal gewachte events en unused channels en gewenste minimum slaag (en dus thresshold ook),
            // TODO Simuleer 10000 keer: bereken P(#FN | #used) uit tabel voor je #FN kandidaten, voor degene boven thresshold brute force de combinaties
            // TODO en kijk of je echt slaagpercentage inderdaad boven je minimum ligt. Je zal moeten meerdere hops (event, channel) geven voor je brute force zoals in echt.
            // TODO Plot ook gemiddeld aantal (event, channel)s die je nodig had voor je eerste brute force slaagde (exctly one)
            // TODO doe dan nog je berekeningen met running probability voor connection interval (P(opeenvolgende) VS P(GCD | n kleinste niet 625 <= clock drift)).
            // TODO dan heb je alle tools om je finaal algoritme te programmeren in deduce_params.rs.
            // TODO maak dummy jambler_hal implementatie die elk een globale packet queue kunnen bekijken en simuleer je volledig algoritme erop.
            // TODO PLot dan of het slaagpercentage overeenkomt met kans slagen channel map * kans slagen conn_interval
            // TODO Implementeer dan in jambler_hal (I2C etc...)
            // TODO Doe dan initiele tests (misschien ga je geen met imperfect channel map vinden...) en kijk of het doet zoals verwacht
            // TODO schrijf je thesis. Check of je LL control PDU lengte trick zou kunnen en vermeld in thesis. Jammer profilen en al de rest is future work
            // TODO (mogelijk Juni) Implementeer jammen, connectie volgen ADHV lengte LL control PDUs en access address sniffing met resterende tijd en goesting
        })
    });
}
fn false_negatives<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("false_negatives.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let used_channels = 0..38_u32;

    // TODO is wrong: misclass chances are based on events to wait and are too important!
    // Do a simulation
    let misclassify_chances = vec![0.10_f64, 0.15]; // given by the change not seen within x events
    let wanted_error_rate = vec![0.1_f64];
    const NUMBER_SIM: u32 = 5000;
    let sims = misclassify_chances
        .iter()
        .cartesian_product(wanted_error_rate.iter())
        .map(|(n, e)| (*n, *e, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let sims = sims
        .into_par_iter()
        .map(|(misclassify_chance, alpha, mut rng)| {
            // For every nb used channels we want the number of false positives
            let stats = (2..38_u32)
                .map(|nb_used| {
                    let mut false_negatives = (0..NUMBER_SIM)
                        .map(|_| {
                            // gen random connection
                            let mut connection = BleConnection::new(&mut rng, Some(nb_used as u8));

                            #[derive(Clone)]
                            struct Occ {
                                channel: u8,
                                used: bool
                            }

                            // Get a random channels sequence to do
                            let mut channels = connection.chm.to_vec().into_iter().enumerate().map(|(c, used)| Occ{ channel: c as u8, used}).collect_vec(); // (channel, is_used)
                            channels.shuffle(&mut rng);
                            
                            // misclass chance geometrical cdf: 1 - (1 - p) ^ (events + 1) => events  = log_{1 - 1 / nb_used}(misclass_chance) - 1
                            let logbase: f64 = (nb_used as f64 - 1.0) / nb_used as f64;
                            let events_to_wait =
                                misclassify_chance.log(logbase).ceil() as u32 - 1;

                            // Check for the used channels if they would be encountered (all short circuits)                            
                            let false_negs = channels.into_iter().filter(|occ| 
                                if occ.used {(0..events_to_wait).all(|_|
                                    connection.next_channel() != occ.channel)} // short circuit first channel occurrence
                                else {(0..events_to_wait).for_each(|_| {connection.next_channel();}); false} // let connection jump events_to_wait
                                ).count();
                            false_negs as f64
                        })
                        .collect_vec();

                    // Map the false positive iter to 1 (var, median) tuple
                    let mean = false_negatives.as_slice().mean();
                    // quantile has no sorting assumed
                    let lower = false_negatives.clone().as_mut_slice().quantile(alpha / 2.0);
                    // upper will select the discreet one just below => add 1 to encompass the complete alpha
                    let upper = false_negatives.as_mut_slice().quantile(1.0 - alpha / 2.0) + 1.0;
                    //println!("{:?}", (mean, (lower, upper)));
                    (nb_used, (mean, (lower, upper)))
                })
                .collect_vec();
            // from it, calculate
            (misclassify_chance, alpha, stats)
        })
        .collect::<Vec<_>>();

    // TODO plot nb_used against its median and var and use miscalssify chance as label

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
        .caption(format!("Expected false negatives for constant error rate ({} simulation) #combinations -> misleading!", NUMBER_SIM), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(used_channels, 1), 0.0..38.0)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("False negatives")
        .x_desc("Used channels")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw theoretical
    for (idx, (misclass_chance, alpha, _)) in sims.iter().enumerate() {
        // new (N, r, n) TODO probleem: r
        //let dist = Hypergeometric::new(nb_used, successes, draws);

        let color = Palette99::pick(idx);

        //let u = *nb_unused;
        let stats = (2..38)
            .map(|i| {
                (
                    i, // TODO nb_unused seems to be nb_used?
                    false_neg_med_conf(i as u8, *misclass_chance, *alpha),
                )
            })
            .collect_vec();

        //stats.iter().for_each(|s| println!("{:?}", s));

        // x range is nb used channels
        let mean = PointSeries::of_element(
            stats.iter().map(|(i, s)| (*i, s.0)),
            5,
            color.to_rgba().filled(),
            &{
                move |coord, size, style| {
                    EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style)   + Text::new("", (0, 15), ("sans-serif", 8))
                }
            },
        );
        let lower_conf = PointSeries::of_element(
            stats.iter().map(|(i, s)| (*i, s.1 .0 as f64)),
            5,
            color.mix(0.5).filled(),
            &{
                move |coord, size, style| {
                    EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style)   + Text::new("", (0, 15), ("sans-serif", 8))
                }
            },
        );
        let upper_conf = PointSeries::of_element(
            stats.iter().map(|(i, s)| (*i, s.1 .1 as f64)),
            5,
            color.mix(0.5).filled(),
            &{
                move |coord, size, style| {
                    EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style)   + Text::new("", (0, 15), ("sans-serif", 8))
                }
            },
        );

        //events_chart.draw_series(
        //    l).unwrap()
        //.label(format!("{} unused theoretical", nb_unused))
        //.legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
        events_chart.draw_series(mean).unwrap();
        events_chart.draw_series(lower_conf).unwrap();
        events_chart.draw_series(upper_conf).unwrap();
    }

    // Draw observed
    for (idx, (misclass_chance, error_rate, mut stats)) in sims.into_iter().enumerate() {
        let color = Palette99::pick(idx);

        //let u = *nb_unused;
        // stats should be in order
        //let iter_base = events_chart
        //    .x_range()
        //    .filter_map(|i| Some((37 - i, stats.get(i as usize)?)));

        stats.sort_by_key(|x| x.0);
        let iter_base = stats.into_iter();

        //println!("OBSERVED");
        
        //iter_base.clone().for_each(|s| println!("{:?}", s));

        // x range is nb used channels
        let mean = LineSeries::new(
            iter_base.clone().map(|(i, y)| (i, y.0)),
            color.to_rgba().filled(),
        );
        let lower_conf = LineSeries::new(
            iter_base.clone().map(|(i, y)| (i, y.1 .0)),
            color.mix(0.5).filled(),
        );
        let upper_conf = LineSeries::new(
            iter_base.clone().map(|(i, y)| (i, y.1 .1)),
            color.mix(0.5).filled(),
        );

        //events_chart.draw_series(
        //    l).unwrap()
        //.label(format!("{} unused theoretical", nb_unused))
        //.legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
        events_chart
            .draw_series(mean)
            .unwrap()
            .label(format!(
                "{:.2} P(missclass.) {:.2} alpha",
                misclass_chance, error_rate
            ))
            .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], color.filled()));
        events_chart.draw_series(lower_conf).unwrap();
        events_chart.draw_series(upper_conf).unwrap();
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

/// (median, (conf-interval))
/// nb_used: number of real unused channels
/// misclassify_chance: the chance to misclassify
/// alpha: 1 - alpha % of samples will be within the returned confidence interval
fn false_neg_med_conf(nb_used: u8, misclassify_chance: f64, alpha: f64) -> (f64, (u8, u8)) {
    // Every used channel has a $(missclassify_chance) chance to be a false negative (=unused)
    let false_neg_dist = Binomial::new(misclassify_chance, nb_used as u64).unwrap();
    let mean = false_neg_dist.mean();
    let l = alpha / 2.0;
    let r = 1.0 - alpha / 2.0;
    //let le = ((1.0-l).ln()/(1.0-misclassify_chance).ln()).ceil() as u8;
    //let re = ((1.0-r).ln()/(1.0-misclassify_chance).ln()).ceil() as u8;
    let left = (0..=37_u8)
        .filter(|g| false_neg_dist.cdf(*g as f64 - 0.5) <= l)
        .max()
        .unwrap_or_else(|| panic!("on params {}", nb_used));
    let right = (0..=37_u8)
        .filter(|g| false_neg_dist.cdf(*g as f64 - 0.5) < r)
        .max()
        .unwrap_or_else(|| panic!("on params {}", nb_used))
        + 1;
    (mean, (left, right))
}

/// Number of extra brute forces next to normal event counter brute force necessary
fn nb_extra_brute_forces(nb_unused: u8, misclassify_chance: f64, alpha: f64) -> u32 {
    // Get conf interval
    let (_, (min_fns, max_fns)) = false_neg_med_conf(nb_unused, misclassify_chance, alpha);
    // For every nb in conf interval, get the number of permutation and sum them all
    (min_fns..=max_fns)
        .map(|fns| binomial(nb_unused as u64, fns as u64))
        .sum::<f64>()
        .round() as u32
}




fn misclass_chance_vs_combinations<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    let mut file_path = params.output_dir;
    file_path.push("misclass_chance_vs_combinations.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let events_to_wait = vec![50, 100, 200];

    let mut dists =  Vec::new();
    for nb_used in 2..=37u8 {
        dists.push(Geometric::new(1.0 / nb_used as f64).unwrap());
    }

    
    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    //let root_area = root_area.margin(20, 20, 20, 20);

    // Turn it into higher level chart
    let mut events_chart = ChartBuilder::on(&root_area)
        // enables Y axis, the size is 40 px in width
        .set_label_area_size(LabelAreaPosition::Left, 120)
        // enable X axis, the size is 40 px in height
        .set_label_area_size(LabelAreaPosition::Bottom, 60) // Reserves place for axis description so they dont collide with labels/ticks
        // Set title. Font and size.
        .caption("Possible combinations (0.10 error, 0.05 alpha) VS false negative chance", ("sans-serif", 20))
        .margin(20)
        .right_y_label_area_size(80)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(0..38u32, (0..0xF_FF_FF_FFu64).log_scale())
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.")
        .set_secondary_coord(0..38u32, 0.0..1.0);

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Possible combinations (log scale)")
        .x_desc("Used channels")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    events_chart
        .configure_secondary_axes()
        .y_desc("Misclassification chance")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();
    
    let v = events_to_wait.len();

    // Draw theoretical
    for (idx, events) in events_to_wait.into_iter().enumerate() {
        
        // information capture: for the same amount of events the misclasify chance is much much lower
        // Maybe still save the method this way

        let misclass_chances =  dists.iter().map(|d| 1.0 - d.cdf(events as f64)).collect_vec();



        let color = Palette99::pick(idx);


        events_chart.draw_secondary_series(LineSeries::new(
            misclass_chances.iter().enumerate().map(|(i, c)| ((i + 2) as u32, *c)), color.mix(0.7).stroke_width(3))).unwrap()
            .label(format!("{} events", events))
            .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
    }




    let color = Palette99::pick(v).to_rgba();

    // x range is nb used channels
    let nb_extra = LineSeries::new(
        (0..38u8).map(|c| (c as u32, nb_extra_brute_forces(37 - c, 0.1, 0.05) as u64)),
        color.stroke_width(2));


    events_chart.draw_series(nb_extra).unwrap()
        .label("#Combinations")
        .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));


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



fn nb_brute_force<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    let mut file_path = params.output_dir;
    file_path.push("nb_brute_force.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let events_to_wait = vec![50, 100, 200];
    let wanted_error_rate = vec![0.1_f64];

    let used_channels = 0..38_u32;
    
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
        .caption("Distribution of number of events until first occurrence for samples per #unused channels", ("sans-serif", 20))
        .margin(20)
        .right_y_label_area_size(80)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(used_channels, 1), 0..1024_u32)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.")
        .set_secondary_coord(0..38u32, 0.0..1.0);

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Times brute force")
        .x_desc("Used channels")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    events_chart
        .configure_secondary_axes()
        .y_desc("misclass chance")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();
    
    // Get the dist of every amount of used channels
    // Geometric: when first time success given p, with p chance of capture = for now 1/nb_used channels
    let mut dists =  Vec::new();
    for nb_used in 2..=37u8 {
        dists.push(Geometric::new(1.0 / nb_used as f64).unwrap());
    }

    // Draw theoretical
    for (idx, (events, alpha)) in events_to_wait.into_iter().cartesian_product(wanted_error_rate.into_iter()).enumerate() {
        
        // information capture: for the same amount of events the misclasify chance is much much lower
        // Maybe still save the method this way

        let misclass_chances =  dists.iter().map(|d| 1.0 - d.cdf(events as f64)).collect_vec();



        let color = Palette99::pick(idx);


        events_chart.draw_secondary_series(LineSeries::new(
            misclass_chances.iter().enumerate().map(|(i, c)| ((i + 2) as u32, *c)), color.mix(0.7).stroke_width(3))).unwrap();


        //let u = *nb_unused;
        let bfs = misclass_chances.into_iter().enumerate()
            .map(|(n,mc)| {
                    nb_extra_brute_forces((37 - (n + 2)) as u8, mc, alpha)
            })
            .collect_vec();


        // x range is nb used channels
        let nb_extra = LineSeries::new(
            bfs.into_iter().enumerate().map(|(i, s)| ((i + 2) as u32, s)),
            color.to_rgba().filled());

        events_chart.draw_series(nb_extra).unwrap()
        .label(format!("{} events {:.2} alpha", events, alpha))
        .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));

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




fn computing_vs_events<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    let mut file_path = params.output_dir;
    file_path.push("computing_vs_events.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let wanted_error_rate = vec![0.1_f64, 0.2, 0.05];

    let events = 50..300_u32;
    
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
        .caption("Given alpha: computing Vs waiting (events)", ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(events.clone(), 0..1024_u32)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Maximum brute force times")
        .x_desc("Events to wait")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Get the dist of every amount of used channels
    // Geometric: when first time success given p, with p chance of capture = for now 1/nb_used channels
    let mut dists =  Vec::new();
    for nb_used in 2..=37u8 {
        dists.push((nb_used, Geometric::new(1.0 / nb_used as f64).unwrap()));
    }

    // Draw theoretical
    for (idx, alpha) in wanted_error_rate.into_iter().enumerate() {
        
        // information capture: for the same amount of events the misclasify chance is much much lower
        // Maybe still save the method this way

        // For all number of events to wait
        let maxes = events.clone().map( |nb_events| {
            // For all channels, get the max
            let m = dists.iter().map( |(nb_used, dist)| {
                let missclass_chance = 1.0 - dist.cdf(nb_events as f64);
                nb_extra_brute_forces(37 - *nb_used, missclass_chance, alpha)
            }).max().unwrap();
            (nb_events, m)
        }).collect_vec();
        

        let color = Palette99::pick(idx);


        // x range is nb used channels
        let nb_extra = LineSeries::new(
            maxes.into_iter().map(|(events, m)| (events, m)),
            color.to_rgba().filled());

        events_chart.draw_series(nb_extra).unwrap()
        .label(format!("{:.2} alpha", alpha))
        .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));

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

// TODO: but which ones? Replot first plot with expected number wrong and it will give us?


/// Given the number of events to wait and alpha, how many would I have to brute force for every amount of used channels?
/// That means, in reality I choose the number of events to wait (and alpha but no given is result nb failures), if I am given 
/// a number of used channels, how many do I have to brute force?
/// In reality, use the given confidence interval to brute force all combinations over the unused channels, for all nb_false_positives in the
/// confidence interval for the false positives
fn how_many_to_check<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("how_many_to_check.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let used_channels = 0..38_u32;

    // TODO is wrong: misclass chances are based on events to wait and are too important!
    // Do a simulation
    let events_to_wait = vec![100_u32]; // given by the change not seen within x events
    let wanted_error_rate = vec![0.1_f64];
    const NUMBER_SIM: u32 = 1000;
    let sims = events_to_wait
        .iter()
        .cartesian_product(wanted_error_rate.iter())
        .map(|(n, e)| (*n, *e, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let sims : Vec<(u32, f64, Vec<(u32, u32)>)> = sims
        .into_par_iter()
        .map(|(events, alpha, mut rng)| {
            // For every nb used channels we want the number of false positives
            let nb_to_brute : Vec<(u32, u32)> = used_channels
                .clone()
                .into_iter()
                .map(|nb_used| {
                    if nb_used <= 1 {
                        // no used channels is no false negatives (logbase would be division by zero)
                        return (0,0);
                    }
                    let mut false_negatives = (0..NUMBER_SIM)
                        .map(|_| {
                            // Get random channels
                            let mut unused_channels = 0;
                            let mut channel_map: u64 = 0x1F_FF_FF_FF_FF;
                            let mut mask;
                            let mut actual_channel_map_seen = [true; 37];
                            while unused_channels < 37 - nb_used {
                                let un = (rng.next_u32() as u8) % 37;
                                mask = 1 << un;
                                if channel_map & mask != 0 {
                                    channel_map ^= mask;
                                    actual_channel_map_seen[un as usize] = false;
                                    unused_channels += 1;
                                }
                            }
                            let mut channel = (rng.next_u32() as u8) % 37;
                            while channel_map & (1 << channel) == 0 {
                                channel = (rng.next_u32() as u8) % 37;
                            }
                            let mut access_address = rng.next_u32();
                            while !is_valid_aa(access_address, jambler::jambler::BlePhy::Uncoded1M)
                            {
                                access_address = rng.next_u32();
                            }
                            let channel_id = calculate_channel_identifier(access_address);
                            let mut event_counter = rng.next_u32() as u16;
                            let mut counter;
                            let chm_info = generate_channel_map_arrays(channel_map);

                            // Simulate for all channels in random order
                            let mut channel_map_seen = [false; 37];
                            let mut channels = actual_channel_map_seen
                                .iter()
                                .enumerate()
                                .filter_map(
                                    |(channel, used)| {
                                        if *used {
                                            Some(channel as u8)
                                        } else {
                                            None
                                        }
                                    },
                                )
                                .collect_vec();
                            channels.shuffle(&mut rng);
                            let mut cur_channel = csa2_no_subevent(
                                event_counter as u32,
                                channel_id as u32,
                                &chm_info.0,
                                &chm_info.1,
                                nb_used as u8,
                            );


                            for channel in channels {
                                // Listen on a channel. Decide unused after events_to_wait events
                                counter = 1_u32;
                                while channel != cur_channel && counter <= events {
                                    counter += 1;
                                    event_counter = event_counter.wrapping_add(1);
                                    cur_channel = csa2_no_subevent(
                                        event_counter as u32,
                                        channel_id as u32,
                                        &chm_info.0,
                                        &chm_info.1,
                                        nb_used as u8,
                                    );
                                }
                                if channel == cur_channel {
                                    channel_map_seen[channel as usize] = true;
                                }
                            }

                            let false_negs = channel_map_seen
                                .iter()
                                .zip(actual_channel_map_seen.iter())
                                .filter(|(observed, actual)| {
                                    if !(**actual) {
                                        assert!(!(**observed))
                                    }
                                    *observed != *actual
                                });
                            false_negs.count() as f64
                        })
                        .collect_vec();

                    // quantile has no sorting assumed
                    let lower = false_negatives.as_mut_slice().quantile(alpha / 2.0) as u32;
                    // upper will select the discreet one just below => add 1 to encompass the complete alpha
                    let upper = (false_negatives.as_mut_slice().quantile(1.0 - alpha / 2.0) + 1.0) as u32;
                    (nb_used, upper - lower)
                })
                .collect_vec();
            // from it, calculate
            (events, alpha, nb_to_brute)
        })
        .collect::<Vec<_>>();


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
        .caption(format!("Expected false negatives for constant error rate ({} simulation) #combinations -> misleading!", NUMBER_SIM), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(used_channels, 1), 0..12_u32)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("How many to check")
        .x_desc("Used channels")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw theoretical
    for (idx, (events, alpha, _)) in sims.iter().enumerate() {
        // new (N, r, n) TODO probleem: r
        //let dist = Hypergeometric::new(nb_used, successes, draws);

        let color = Palette99::pick(idx);

        let iter_plot = (2..38_u32).map(|nb_used|
        {
            let dist = Geometric::new(1.0 / nb_used as f64).unwrap();
            let missclass_chance = 1.0 - dist.cdf(*events as f64);
            let (_mean, (lower, upper)) = false_neg_med_conf((37 - nb_used) as u8, missclass_chance, *alpha);
            (nb_used, (upper - lower))
        }
        ).collect_vec();

        let lower_conf = PointSeries::of_element(
            iter_plot.iter().map(|(i, s)| (*i, *s as u32)),
            5,
            color.mix(0.5).filled(),
            &{
                move |coord, size, style| {
                    EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style)   + Text::new("", (0, 15), ("sans-serif", 8))
                }
            },
        );

        events_chart.draw_series(lower_conf).unwrap();
    }

    // Draw observed
    for (idx, (events, alpha, nb_to_brute)) in sims.into_iter().enumerate() {
        let color = Palette99::pick(idx);

        
        let br = LineSeries::new(
            nb_to_brute.into_iter(),
            color.to_rgba().filled(),
        );

        events_chart
            .draw_series(br)
            .unwrap()
            .label(format!(
                "{:.2} events {:.2} error rate",
                events, alpha
            ))
            .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], color.filled()));
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

// TODO do an actual simulation (give nb unused etc) to see if you indeed get the alpha as predicted