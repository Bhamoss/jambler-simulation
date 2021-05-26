use itertools::Itertools;
use ordered_float::OrderedFloat;
use plotters::prelude::*;
use rand::{Rng, seq::SliceRandom};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{fs::{create_dir_all, File}};

use statrs::distribution::{Binomial, Discrete, Geometric};
use statrs::statistics::OrderStatistics;
use statrs::{distribution::Univariate, statistics::Mean};
use std::f64;
use crate::csa2::csa2_no_subevent_unmapped;
use num::{integer::{binomial}};

use crate::{SimulationParameters, Task, run_tasks, tasks::BleConnection};
use crate::csa2::{csa2_no_subevent, generate_channel_map_arrays};


use std::sync::{Arc, Mutex};
use indicatif::{MultiProgress};
// use indicatif::{ ProgressBar, ProgressStyle};

// TODO morgen: JE BENT ER BIJNA!!!!!
// TODO Doe bovenstaande als simulatie: voor de nb_used = 37 - *nb_unused_seen + nb_false_negs
// TODO Simuleer 1000 keer en kijk hoeveel keer (relatief)  hij nb_false_negs als false negatives had (used als unused).
// TODO Check of dit dan overeenstemt -> JAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
// TODO Plot dan voor een gegeven aantal unused en events het equivalent van je tabel: 2-y axissen met op ene P(#FN | #used) en andere #Combinations -> JAAAAA
// TODO Plot dan (mss eerst tabel) voor een gegeven selectie percentage (10% van hierboven), de som van de kansen voor #FNs die daarboven liggen voor ieder aantal used channels.
// TODO Selecteer daar een verstandig minimum uit (je worst case slaag percentage) --> JAAAAAAAAAAAAAAAAAAAAAAAAAAA (nog niet gekozen, hangt af van brute force aantallen)
// TODO Plot/tabel dan voor gegeven aantal events, hoeveel burte forces (som #Combos boven 10%) je moet doen voor iedere mogelijk aantal geziene unused channels
// TODO Hopelijk kan je dan een minimum slaag percentage selecteren waarvoor de maximum brute force over alle unused channels computationally feasable is. -> JAAAAAAAAAAAAAAAAAAAAAA
// TODO Herdoe/ pas aan met fysiek capture percentage. -> JAAAAAAAAAAAAAAAAAAAAAAAAAAA (maar ben niet zo heel zeker, volgende TODO zal data aantonen)
// het volgende kan niet omdat je #events er ook toe doen: Om het aft te maken, plot dan ook je minimum slaag percentage op y as naar het maximum brute force (over gezien unused) (slaag percentage is NIET thresshold percentage).
// De redenering is: maximum brute force voor je compute power, dan kijken minimum error, dan #events daarop plotten

// TODO Dan SIMULEER: gegeven physcial chance en max brute force en max error, bereken aantal gewachte events en unused channels en gewenste minimum slaag (en dus thresshold ook),
// TODO plot de boxplots error rate per (ongekende) aantal echte used channels EN hoeveel brute forces je gedaan hebt voor ieder observed aantal unused channels (lijn)
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

// TODO do the P(#FN | #Used) have to sum to 1 for an unused/events couple? 
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


pub fn channel_recovery<R: RngCore + Send + Sync>(mut params: SimulationParameters<R>, bars: Arc<Mutex<MultiProgress>>) {
    params.output_dir.push("channel_recovery");
    create_dir_all(&params.output_dir).unwrap();
    let tasks: Vec<Box<dyn Task>> = vec![
        Box::new(false_negatives),
        Box::new(false_negative_chance_vs_combinations),
        Box::new(thressholds),
        // TODO uncomment, takes to much time
        Box::new(with_capture_chance),
        Box::new(chm_sim),
    ]; // chm_sim
    run_tasks(tasks, params, bars);
    println!("Channel recovery done");
}
#[derive(Clone)]
struct Occ {
    channel: u8,
    used: bool
}

fn chance_and_combo(nb_unused_seen: u8, nb_false_negs: u8, nb_events: u8) -> (f64, u64) {
    let nb_used = 37 - nb_unused_seen + nb_false_negs;
    // So now I waited everywhere for nb_evenst events and found nb_unused_seen unused channels (with possible false negatives)
    // Assume nb_false_negs false negatives and thus nb_used actual channels under that assumptions.
    // What is the chance of this occurring?
    let dist = Geometric::new(1.0 / nb_used as f64).unwrap(); // pdf(e): chance of first occurring at e
    let chance_of_hearing_channel = dist.cdf(nb_events as f64); // cdf(e): chance of first occurring on or before e
    let all_dist = Binomial::new(chance_of_hearing_channel, nb_used as u64).unwrap(); // pdf(e) = chance of hearing exactly e of nb_used channels
    let prob = all_dist.pmf((nb_used - nb_false_negs) as u64); // the chance of seeing exactly that many true positives (and thus false negatives)
    // The chance of the true positives (no false positives possible)
    //let mut prob = dist.cdf(nb_evenst as f64).powi(nb_used.into()); // cdf(e): chance of first occurring on or before e
    // The chance of the number of false negatives
    //prob *= (1.0 - dist.cdf(nb_evenst as f64)).powi(nb_false_negs.into()); // 1.0 - cdf(e): chance of not occurring on or before e
    // The true negatives have a probability of 1 for not occurring and a probability of 0 for occurring
    let nb_bf = binomial(nb_unused_seen as u64, nb_false_negs as u64) as u64;
    (prob, nb_bf)
}


fn chance_and_combo_reality(nb_unused_seen: u8, nb_false_negs: u8, nb_events: u8, physical_chance: f64) -> (f64, u64) {
    let nb_used = 37 - nb_unused_seen + nb_false_negs;
    // So now I waited everywhere for nb_evenst events and found nb_unused_seen unused channels (with possible false negatives)
    // Assume nb_false_negs false negatives and thus nb_used actual channels under that assumptions.
    // What is the chance of this occurring?
    let real_capture_chance = physical_chance * (1.0 / nb_used as f64);
    let dist = Geometric::new(real_capture_chance).unwrap(); // pdf(e): chance of first occurring at e
    let chance_of_hearing_channel = dist.cdf(nb_events as f64); // cdf(e): chance of first occurring on or before e
    let all_dist = Binomial::new(chance_of_hearing_channel, nb_used as u64).unwrap(); // pdf(e) = chance of hearing exactly e of nb_used channels
    let prob = all_dist.pmf((nb_used - nb_false_negs) as u64); // the chance of seeing exactly that many true positives (and thus false negatives)
    // The chance of the true positives (no false positives possible)
    //let mut prob = dist.cdf(nb_evenst as f64).powi(nb_used.into()); // cdf(e): chance of first occurring on or before e
    // The chance of the number of false negatives
    //prob *= (1.0 - dist.cdf(nb_evenst as f64)).powi(nb_false_negs.into()); // 1.0 - cdf(e): chance of not occurring on or before e
    // The true negatives have a probability of 1 for not occurring and a probability of 0 for occurring
    let nb_bf = binomial(nb_unused_seen as u64, nb_false_negs as u64) as u64;
    (prob, nb_bf)
}


fn false_negative_chance_vs_combinations<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    let mut together_path = file_path.clone();
    together_path.push("false_negative_chance_VS_combinations.png");

    File::create(together_path.clone()).expect("Failed to create plot file");

    // Dir for per nb unused observed
    file_path.push("false_negative_chance_VS_combinations");
    create_dir_all(file_path.clone()).expect("Failed to create plot directory");

    // First one gets taken as the number of events to plot them all together
    let events = vec![100_u8, 70, 150];
    // Configure in filter which ones get used in together using module (now %2)
    //let nb_unused_observed = vec![2_u8,8, 14,25,26];
    let nb_unused_observed = (0u8..37).collect_vec();
    let sims = nb_unused_observed.into_iter().map(|n| (n, ChaCha20Rng::seed_from_u64(rng.next_u64())))
    .collect_vec();
    let data = sims.into_par_iter().map(|(nb_unused_seen, mut urng)| {
        let sims = events.iter().map(|n| (*n, ChaCha20Rng::seed_from_u64(urng.next_u64())))
        .collect_vec();
        let nu_data = sims.into_par_iter().map(|(nb_evenst, mut rng)| {
            let probs = (0..=nb_unused_seen).map(|nb_false_negs| {
                let nb_used = 37 - nb_unused_seen + nb_false_negs;
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
                let nb_bf = binomial(nb_unused_seen as u64, nb_false_negs as u64) as u64;
                let (probf, nb_bff) = chance_and_combo(nb_unused_seen, nb_false_negs, nb_evenst);
                assert_eq!(nb_bf, nb_bff);
                assert!((prob - probf).abs() < 0.01);

                // SIM
                const NUMBER_SIM: u32 = 1000; // TODO up to 10_000
                let false_negatives_seen = (0..NUMBER_SIM)
                        .map(|_| {
                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, Some(nb_used as u8));
                    // Get a random channels sequence to do
                    let mut channels = connection.chm.to_vec().into_iter().enumerate().map(|(c, used)| Occ{ channel: c as u8, used}).collect_vec(); // (channel, is_used)
                    channels.shuffle(&mut rng);
                    // Check for the used channels if they would be encountered (all short circuits)                            
                    let false_negs = channels.into_iter().filter(|occ| 
                        if occ.used {(0..nb_evenst).all(|_|
                            connection.next_channel() != occ.channel)} // short circuit first channel occurrence
                        else {(0..nb_evenst).for_each(|_| {connection.next_channel();}); false} // let connection jump events_to_wait
                        ).count();
                    false_negs as u8
                }).filter(|fns_o| *fns_o == nb_false_negs).count() as f64 / NUMBER_SIM as f64;


                (nb_false_negs, nb_used, chance_of_hearing_channel, prob, false_negatives_seen, nb_bf)
            }).collect_vec();

            /*
            println!("Observed {} unused when waiting for {} events:\n", nb_unused_seen, nb_evenst);
            let s = "| #FN | #used | P(capture) | P(#FN | #used) | SimFreq | #Combinations";
            println!("{}",s);
            s.chars().for_each(|_| print!("-"));
            println!();
            probs.iter().for_each(|p| println!("| {:3} | {:5} | {:9.2}% | {:13.2}% | {:7.2}% | {}", p.0, p.1, p.2 * 100.0, p.3 * 100.0, p.4 * 100.0, p.5));
            //println!("Sum: {}", probs.into_iter().map(|p| p.3).sum::<f64>());
            println!("\n\n\n");
            */
            (nb_evenst, probs)
        }).collect::<Vec<_>>();
    (nb_unused_seen, nu_data)
    }).collect::<Vec<_>>();

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920
                             // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(together_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();


    let mut events_chart = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 120)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption(format!("P(#False negatives | #Used) VS #Combinations ({} events)", data.first().unwrap().1.first().unwrap().0), ("sans-serif", 20))
        .margin(20)
        .right_y_label_area_size(80)
        .build_cartesian_2d(0..38u32, 0.0..1.05f64)
        .expect("Chart building failed.")
        .set_secondary_coord(0..38u32, (1..10_000_000u64).log_scale());
    events_chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("P(#False negatives | #Used)")
        .x_desc("#False negatives")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();
    events_chart
        .configure_secondary_axes()
        .y_desc("Possible combinations (log scale)")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();
    for (idx, (nb_unused, nu_data)) in data.iter().filter(|(nb_unused, _)| *nb_unused % 7 == 0).enumerate() {
        let (_nb_events, probs) = nu_data.first().unwrap();
        let color = Palette99::pick(idx);
        let cfull = color.to_rgba();
        let cmix = color.mix(0.7);
        // Draw theoretical
        let txt =  |coord : (u32, f64), size : i32, style: ShapeStyle| {
            EmptyElement::at(coord) 
            + Circle::new((0, 0), size, style)   
            + Text::new(if coord.1 > 0.10 {(*nb_unused as u32 - coord.0).to_string()} else {"".to_string()}, (10, -10), ("sans-serif", 10))
        };
        let t = PointSeries::of_element(
            probs.iter().map(|p| (p.0 as u32, p.3)),
            5,
            cfull.clone().filled(),
            &txt,
        );
        let cfull = color.to_rgba();
        events_chart.draw_series(t).unwrap()
        .label(format!("{} observed unused", nb_unused))
        .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], cfull.filled()));
        //let cfull = color.to_rgba();
        // Draw observed
        let o = LineSeries::new(
            probs.iter().map(|p| (p.0 as u32, p.4)),
            cmix.stroke_width(3));
        events_chart.draw_series(o).unwrap();
        // Draw combinations
        let cfull = color.to_rgba();
        let c = LineSeries::new(
            probs.iter().filter(|p| p.0 >= 2).map(|p| (p.0 as u32, p.5)), cfull.stroke_width(3));
        events_chart.draw_secondary_series(c).unwrap();
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


    // Draw seperately, all events for all
    data.into_par_iter().for_each(|(nb_unused, un_data)| {
    //for (nb_unused, un_data) in data.into_iter() { file_error_path
        let mut this_file = file_path.clone(); 
        this_file.push(format!("false_negative_chance_VS_combinations_{}_unused.png", nb_unused));
        File::create(this_file.clone()).expect("Failed to create plot file");

        let root_area = BitMapBackend::new(this_file.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();
        let mut this_unused_chart = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 120)
            .set_label_area_size(LabelAreaPosition::Bottom, 60)
            .caption(format!("{} observed unused: P(#False negatives | #Used) VS #Combinations", nb_unused).as_str(), ("sans-serif", 20))
            .margin(20)
            .right_y_label_area_size(80)
            .top_x_label_area_size(50)
            .build_cartesian_2d(0..38u32, 0.0..1.05f64)
            .expect("Chart building failed.")
            .set_secondary_coord(0..38u32, (1..10_000_000u64).log_scale()); // Leave all on this scale to better compare each picture
        this_unused_chart
            .configure_mesh()
            .disable_x_mesh()
            .y_desc("P(#False negatives | #Used)")
            .x_desc("#False negatives")
            .label_style(("sans-serif", 20)) // The style of the numbers on an axis
            .axis_desc_style(("sans-serif", 20))
            .draw()
            .unwrap();
        let nb_used_formatter = |fns_x:&u32 | if nb_unused as i32 - *fns_x as i32 >= 0 {(37 - nb_unused + *fns_x as u8).to_string()} else {"".to_string()} ;
        this_unused_chart
            .configure_secondary_axes()
            .y_desc("Possible combinations (log scale)")
            .x_desc("#Used under FN assumption")
            .x_label_formatter(&nb_used_formatter)
            .label_style(("sans-serif", 20)) // The style of the numbers on an axis
            .axis_desc_style(("sans-serif", 20))
            .draw().unwrap();

        // Draw combinations
        let cfull = Palette99::pick(0).to_rgba();
        let c = LineSeries::new(
            un_data.first().unwrap().1.iter().map(|p| (p.0 as u32, p.5)), cfull.stroke_width(3));
        this_unused_chart.draw_secondary_series(c).unwrap()
        .label("#Combinations")
        .legend( move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], cfull.filled()));

        for (idx, (nb_events, probs)) in un_data.into_iter().enumerate() {
            let color = Palette99::pick(idx + 1);
            let cfull = color.to_rgba();
            let cmix = color.mix(0.5);
            // Draw theoretical
            let txt =  |coord : (u32, f64), size : i32, style: ShapeStyle| {
                EmptyElement::at(coord) // have text and circle be relative to this
                + Circle::new((0, 0), size, style)   
            };
            let t = PointSeries::of_element(
                probs.iter().map(|p| (p.0 as u32, p.3)),
                5,
                cfull.clone().filled(),
                &txt,
            );
            let cfull = color.to_rgba();
            this_unused_chart.draw_series(t).unwrap()
            .label(format!("{} events", nb_events))
            .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], cfull.filled()));
            // Draw observed
            let o =LineSeries::new(
                probs.iter().map(|p| (p.0 as u32, p.4)),
                cmix.stroke_width(3));
            this_unused_chart.draw_series(o).unwrap();
        }
        // Draws the legend
        this_unused_chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 15))
            .position(SeriesLabelPosition::UpperRight)
            .draw()
            .unwrap();
    });
}
fn false_negatives<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("false_negatives.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    let used_channels = 0..38_u32;

    // Do a simulation
    let misclassify_chances = vec![0.10_f64, 0.15]; // given by the change not seen within x events
    let wanted_error_rate = vec![0.1_f64];
    const NUMBER_SIM: u32 = 1000; // TODO Up to 5000
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




fn thressholds<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    let file_path = params.output_dir;
    let mut rng = params.rng;
    let mut together_error_path = file_path.clone();
    together_error_path.push("error_dist.png");
    File::create(together_error_path.clone()).expect("Failed to create plot file");
    let mut file_error_path = file_path;
    file_error_path.push("error_dists");
    create_dir_all(file_error_path.clone()).expect("Failed to create plot directory");


    let thressholds = vec![0.05f64, 0.1, 0.2];
    let events = vec![70u8, 100, 150];
    let es = 50u32..200;
    let sims = (1u8..=37)
        .map(|n| (n, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let k = sims.into_par_iter().map(|(used, mut rng)| {
        let d = events.clone().into_iter().map(|nb_evenst| {
            // SIM
            const NUMBER_SIM: u32 = 1000; // TODO up to 10_000
            let false_negatives_seen_sim = (0..NUMBER_SIM).map(|_| {
                // gen random connection
                let mut connection = BleConnection::new(&mut rng, Some(used));
                // Get a random channels sequence to do
                let mut channels = connection.chm.to_vec().into_iter().enumerate().map(|(c, used)| Occ{ channel: c as u8, used}).collect_vec(); // (channel, is_used)
                channels.shuffle(&mut rng);
                // Check for the used channels if they would be encountered (all short circuits)                            
                let false_negs = channels.into_iter().filter(|occ| 
                    if occ.used {(0..nb_evenst).all(|_|
                        connection.next_channel() != occ.channel)} // short circuit first channel occurrence
                    else {(0..nb_evenst).for_each(|_| {connection.next_channel();}); false} // let connection jump events_to_wait
                    ).count();
                false_negs as u8
            }).collect_vec();

            let v = (0u8..=used).map(|fns| (fns, false_negatives_seen_sim.iter().filter(|fns_o| **fns_o == fns).count() as f64 / NUMBER_SIM as f64)).collect_vec();

            (nb_evenst, v)
        }).collect_vec();
        (used, d)
    }).collect::<Vec<_>>();

    const HEIGHT: u32 = 1080;
    const WIDTH: u32 = 1080; // was 1920

    // Draw the error for all number of events. The error is the maximum error for all nb_(un)used, where the error for a number (un)used is  
    // the sum of chances for the possible false negatives for which the chance of their occurrence is below the thresshold 
    let root_area = BitMapBackend::new(together_error_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();


    let mut events_chart = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 120)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .caption("Maximum error and brute forces by #events to wait. BUT worst ones probably most likely! (e.g. 10 unused)", ("sans-serif", 20))
        .margin(20)
        .right_y_label_area_size(80)
        .build_cartesian_2d(es.clone(), 0.0..1.05f64)
        .expect("Chart building failed.")
        .set_secondary_coord(es.clone(), 1..1_000u64); // Lea;
    events_chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Error % (sum under thresshold)")
        .x_desc("Events to wait")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();
    events_chart.configure_secondary_axes()
        .y_desc("Possible combinations")
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw().unwrap();

    
    let l = thressholds.len();
        
    for (idx, thresshold) in thressholds.into_iter().enumerate() {
        let color = Palette99::pick(idx);

        let errors = es.clone().map(|nb_events| {
            // Max over nb_used -> this is the unknown
            let mut err : Vec<f64> = (1u8..=37).map(|nb_used| 
                // Sum all below thresshold for the nb of false negatives possible
                (0u8..=nb_used).map(|nb_false_neg| chance_and_combo(37 -nb_used + nb_false_neg, nb_false_neg,nb_events as u8).0)
                .filter(|fn_chance| *fn_chance < thresshold).sum::<f64>()
            ).collect_vec();
            let q85 = err.as_mut_slice().quantile(0.85);
            let max = err.into_iter().map(OrderedFloat::from).max().unwrap().into_inner();
            (nb_events, max, q85)
        }).collect_vec();

        // Draw 
        //let o = LineSeries::new(
        //    errors.iter().map(|(events, _, q85)| (*events as u32, *q85)),
        //    color.mix(0.6).stroke_width(3));
        //events_chart.draw_series(o).unwrap();
        let o = LineSeries::new(
            errors.iter().map(|(events, max_err, _)| (*events as u32, *max_err)),
            color.stroke_width(3));
        events_chart.draw_series(o).unwrap()
        .label(format!("{:.2} thresshold error", thresshold))
        .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], color.filled()));


        let bfs = es.clone().map(|nb_events| {
            // Max over nb_unused -> this is the observed
            let nbu = (0u8..37).map(|nb_unused_seen| 
                // Sum combo all above thresshold for the nb of false negatives it would be for this nb of unused
                (0u8..=nb_unused_seen).map(|nb_false_neg| chance_and_combo(nb_unused_seen, nb_false_neg,nb_events as u8))
                .filter(|(fn_chance, _)| *fn_chance >= thresshold).map(|(_, bfs)| bfs).sum::<u64>()
            ).collect_vec();
            let q85 = nbu.iter().map(|j| *j as f64).collect_vec().as_mut_slice().quantile(0.85).round() as u64;
            let max = nbu.into_iter().max().unwrap();
            (nb_events, max, q85)
        }).collect_vec();

        // Draw on secondary axis
        let color = Palette99::pick(idx + l);
        //let o = LineSeries::new(
        //    bfs.iter().map(|(events, _, q85)| (*events as u32, *q85)),
        //    color.mix(0.6).stroke_width(3));
        //events_chart.draw_secondary_series(o).unwrap();
        let o = LineSeries::new(
            bfs.iter().map(|(events, max_err, _)| (*events as u32, *max_err)),
            color.stroke_width(3));
        events_chart.draw_secondary_series(o).unwrap()
        .label(format!("{:.2} thresshold #brute force", thresshold))
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


    // Draw for all nb of used their FN distribution
    k.into_par_iter().for_each(|(nb_used, data)| {
        // Draw the error dist for the number of false negatives for this unused
        let mut this_file = file_error_path.clone(); 
        this_file.push(format!("error_dist_{}_used.png", nb_used));
        File::create(this_file.clone()).expect("Failed to create plot file");

        // Draw the error for all number of events. The error is the maximum error for all nb_(un)used, where the error for a number (un)used is  
        // the sum of chances for the possible false negatives for which the chance of their occurrence is below the thresshold 
        let root_area = BitMapBackend::new(this_file.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();


        let mut events_chart = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 120)
            .set_label_area_size(LabelAreaPosition::Bottom, 60)
            .caption(format!("{} used: FN distribution", nb_used), ("sans-serif", 20))
            .margin(20)
            .build_cartesian_2d(0..38u32, 0.0..1.05f64)
            .expect("Chart building failed.");
        events_chart
            .configure_mesh()
            .disable_x_mesh()
            .y_desc("Probability of this #FN occuring")
            .x_desc("Number of false negatives")
            .label_style(("sans-serif", 20)) // The style of the numbers on an axis
            .axis_desc_style(("sans-serif", 20))
            .draw()
            .unwrap();
            
        for (idx, (events, fn_dist)) in data.into_iter().enumerate() {
            let color = Palette99::pick(idx);

            // Draw theoretical
            let t = PointSeries::of_element(
                fn_dist.iter().map(|p| (p.0 as u32, chance_and_combo(37- nb_used + p.0, p.0, events).0)),
                5,
                color.filled(),
                &{|coord, size, style| {
                    EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style)   
                }},
            );
            events_chart.draw_series(t).unwrap();
            // Draw 
            let o = LineSeries::new(
                fn_dist.into_iter().map(|(fns, err)| (fns as u32, err)),
                color.stroke_width(3));
            events_chart.draw_series(o).unwrap()
            .label(format!("{} events", events))
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
        
    });

           
}

fn with_capture_chance<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>)  {

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("with_physical_error");
    create_dir_all(file_path.clone()).expect("Failed to create plot directory");

    let max_bfs_feasable = vec![10u64, 100, 1000];
    let max_error_rates = vec![0.1f64, 0.2, 0.05];

    let plots = max_bfs_feasable.into_iter().cartesian_product(max_error_rates.into_iter())
    .map(|(b, e)| (b, e,ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
    plots.into_par_iter().for_each(|(bfs_max, max_error, mut _rng)| {
        //
        let mut path = file_path.clone();
        path.push(format!("reality_{}bfs_{:.2}err.png",  bfs_max, max_error));
        File::create(path.clone()).expect("Failed to create plot file");


        const HEIGHT: u32 = 1080;
        const WIDTH: u32 = 1080; // was 1920

        // Draw the error for all number of events. The error is the maximum error for all nb_(un)used, where the error for a number (un)used is  
        // the sum of chances for the possible false negatives for which the chance of their occurrence is below the thresshold 
        let root_area = BitMapBackend::new(path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();


        let mut events_chart = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 120)
            .set_label_area_size(LabelAreaPosition::Bottom, 60)
            .caption(format!("reality: {} max feasable brute force {:.2}% max error", bfs_max, max_error), ("sans-serif", 20))
            .margin(20)
            .build_cartesian_2d(0.2..1.05f64, 0..260u32)
            .expect("Chart building failed.");
        events_chart
            .configure_mesh()
            .disable_x_mesh()
            .y_desc("#Events required")
            .x_desc("Physical error chance")
            .label_style(("sans-serif", 20)) // The style of the numbers on an axis
            .axis_desc_style(("sans-serif", 20))
            .draw()
            .unwrap();
        

        let es = 50u8..=255;


        let pcs = (0..80).into_par_iter().map(|i| 0.20 + 0.01 * i as f64);
        let pc_to_smalles_valid_event = pcs.filter_map(|pc| {
            // For the given brute force max, find the lowest number of events for which a thresshold exists, for which the max error is lower than the given one
            // Brute force this for now
            let smallest_valid_nb_events = es.clone().find(|nb_events| {
                // Get allowed threshes
                let threshes = (1..100).map(|i| 0.01 * i as f64);
                let bf_threshes = threshes.skip_while(|thresshold| {
                    // Check if this would go over the BF thresh
                    // Max over nb_unused -> this is the observed
                    let nbu = (0u8..37).map(|nb_unused_seen| 
                        // Sum combo all above thresshold for the nb of false negatives it would be for this nb of unused
                        (0u8..=nb_unused_seen).map(|nb_false_neg| chance_and_combo_reality(nb_unused_seen, nb_false_neg,*nb_events as u8, pc))
                        .filter(|(fn_chance, _)| *fn_chance >= *thresshold).map(|(_, bfs)| bfs).sum::<u64>()
                    ).max();
                    if let Some(bfs) = nbu {
                        //if bfs > bfs_max && *thresshold > 0.1 {
                        //    println!("Too much {:.2} {}", *thresshold, bfs)
                        //}
                        bfs > bfs_max
                    }
                    else {
                        println!("Invalid");
                        false
                    }}
                ).collect_vec();


                // Check if any of these thresses are below the max error
                let found = bf_threshes.into_iter().rev().find(|thresshold| {
                        // Max over nb_used -> this is the unknown
                        let err =(1u8..=37).map(|nb_used| 
                            // Sum all below thresshold for the nb of false negatives possible
                            (0u8..=nb_used).map(|nb_false_neg| chance_and_combo_reality(37 -nb_used + nb_false_neg, nb_false_neg,*nb_events as u8, pc).0)
                            .filter(|fn_chance| *fn_chance < *thresshold).sum::<f64>()
                        ).map(OrderedFloat::from).max();
                        if let Some(err) = err {
                            err.into_inner() <= max_error
                        }
                        else {
                            false
                        }
                });
                found.is_some()
            });
            smallest_valid_nb_events.map(|e| (pc, e))
        }).collect::<Vec<_>>();
        

        let o = LineSeries::new(
            pc_to_smalles_valid_event.iter().map(|(pc, events)| (*pc, *events as u32)),
            RED.to_rgba().stroke_width(3));
        events_chart.draw_series(o).unwrap();

    });
}



fn chm_sim<R: RngCore + Send + Sync>(params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>)  {

    const NUMBER_SIMS : u32 = 500;

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push(format!("chm_sim_{}", NUMBER_SIMS));
    create_dir_all(file_path.clone()).expect("Failed to create plot directory");
    let mut error_path = file_path.clone();
    error_path.push("error");
    create_dir_all(error_path.clone()).expect("Failed to create plot directory");
    let mut bf_path = file_path.clone();
    bf_path.push("number_brute_forces");
    create_dir_all(bf_path.clone()).expect("Failed to create plot directory");
    let mut extra_path = file_path.clone();
    extra_path.push("number_extra_packets");
    create_dir_all(extra_path.clone()).expect("Failed to create plot directory");
    let mut nt_path = file_path.clone();
    nt_path.push("number_total_events");
    create_dir_all(nt_path.clone()).expect("Failed to create plot directory");
    let mut extra_events_path = file_path.clone();
    extra_events_path.push("number_extra_events");
    create_dir_all(extra_events_path.clone()).expect("Failed to create plot directory");
    let mut seperate_fn_path = file_path;
    seperate_fn_path.push("seperate_fn_dists");
    create_dir_all(seperate_fn_path.clone()).expect("Failed to create plot directory");



    let max_bfs_feasable = vec![100, 500];
    let max_error_rates = vec![0.1, 0.3];
    let physical_error_rates = vec![0.1, 0.5];

    //let nt = NUMBER_SIMS as usize * 36 * max_bfs_feasable.len() * max_error_rates.len() * physical_error_rates.len();


    //let todo = Mutex::new((2u8..38).collect_vec());

    let plots = max_bfs_feasable.into_iter().cartesian_product(max_error_rates.into_iter().cartesian_product(physical_error_rates.into_iter()))
    .map(|(b, (e, p))| (b, e, p,ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
    plots.into_par_iter().for_each(|(bfs_max, max_error, packet_loss,mut rng)| {

        // calculate number of events to wait and the thresshold
        let es = 50u8..=255;
        // For the given brute force max, find the lowest number of events for which a thresshold exists, for which the max error is lower than the given one
        // Brute force this for now
        let found = es.filter_map(|nb_events| {
            // Get allowed threshes
            let threshes = (1..100).map(|i| 0.01 * i as f64);
            let bf_threshes = threshes.skip_while(|thresshold| {
                // Check if this would go over the BF thresh
                // Max over nb_unused -> this is the observed
                let nbu = (0u8..37).map(|nb_unused_seen| 
                    // Sum combo all above thresshold for the nb of false negatives it would be for this nb of unused
                    (0u8..=nb_unused_seen).map(|nb_false_neg| chance_and_combo_reality(nb_unused_seen, nb_false_neg,nb_events, 1.0 - packet_loss))
                    .filter(|(fn_chance, _)| *fn_chance >= *thresshold).map(|(_, bfs)| bfs).sum::<u64>()
                ).max();
                if let Some(bfs) = nbu {
                    //if bfs > bfs_max && *thresshold > 0.1 {
                    //    println!("Too much {:.2} {}", *thresshold, bfs)
                    //}
                    bfs > bfs_max
                }
                else {
                    println!("Invalid");
                    false
                }}
            ).collect_vec();

            // Check if any of these thresses are below the max error
            // rev() does not really matter, it just has to have 1 satisfying, the lower the sooner it is found
            let found = bf_threshes.into_iter().rev().find(|thresshold| {
                    // Max over nb_used -> this is the unknown
                    let err =(1u8..=37).map(|nb_used| 
                        // Sum all below thresshold for the nb of false negatives possible
                        (0u8..=nb_used).map(|nb_false_neg| chance_and_combo_reality(37 -nb_used + nb_false_neg, nb_false_neg,nb_events, 1.0 - packet_loss).0)
                        .filter(|fn_chance| *fn_chance < *thresshold).sum::<f64>()
                    ).map(OrderedFloat::from).max();
                    if let Some(err) = err {
                        err.into_inner() <= max_error
                    }
                    else {
                        false
                    }
            });
            found.map(|thress| (nb_events, thress))
        }).next();

        if let Some((events, thress)) = found {

            //println!("Simulating {} bfs {:.2} err {:.} pl with {} events and {:.2} thress",  bfs_max, max_error, packet_loss, events, thress);

            // simulate for every possible real used
            let sims = (2u8..=37).map(|i| (i, ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
            // Contains (actual_nb_used, vec over simulation with ((success, not no_solution), nb_bfs, extra_packets after channel map))
            let sims = sims.into_par_iter().map(|(nb_used, mut rng)|{


                let sims = (0..NUMBER_SIMS).map(|i| (i, ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
                let resses = sims.into_par_iter().map(|(_ns, mut rng)| {
                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, Some(nb_used as u8));
                    // Get a random channels sequence to do
                    let mut channels = (0u8..37).collect_vec(); // (channel, is_used)
                    channels.shuffle(&mut rng);

                    // Simulate the channel map stage     
                    let mut total_nb_events : u64 = 0;                   
                    let mut observerd_packets = channels.into_iter().filter_map(|channel| 
                        (0..events).find_map(|_| {
                            total_nb_events += 1;
                            if connection.next_channel() == channel && rng.gen_range(0.0..1.0) <= 1.0-packet_loss {
                                Some((connection.cur_event_counter, connection.cur_channel))
                            } else {None}
                        })).collect_vec();

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
                        extra_packets += 1;
                        // Get random next channel to listen for extra packet
                        let channel = observed_used[rng.gen_range(0..observed_used.len())];
                        // Listen until you hear one
                        let  next_one = (0..).map(|_| {connection.next_channel(); total_nb_events += 1; (connection.cur_event_counter.wrapping_sub(relative_event_offset), connection.cur_channel)}).find(|c| c.1 == channel  && rng.gen_range(0.0..1.0) <= 1.0-packet_loss).unwrap();
                        // Add to observed
                        observerd_packets.push(next_one);

                        //if nb_used == 37 && extra_packets > 50 {println!("Large packets {} for {}", extra_packets, nb_used)};

                        // Brute force again
                        result = brute_force(extra_packets, bfs_max, nb_used, connection.channel_map, relative_event_offset, observerd_packets.as_slice(), chm, thress, events, packet_loss, connection.channel_id);

                    }

                    let mut extra_events = 0u64;
                    // Follow for unknown ones
                    let r = if let CounterInterval::ExactlyOneSolution(first_packet_actual_counter, chm, chm_todo) = result.0 {
                        if chm_todo == 0 {
                            CounterInterval::ExactlyOneSolution(first_packet_actual_counter, chm, 0)
                        }
                        else{
                            let mut running_chm = chm;
                            let mut running_counter = connection.cur_event_counter; // In reality should calculate with elapsed time and cur time
                            let nb_to_see = geo_qdf(1.0 - packet_loss, 0.98) as u8;
                            let mut todo_seen = [0u8;37];
                            (0..37usize).for_each(|c| {
                                if (1u64 << c) & chm_todo != 0 {
                                    todo_seen[c] = nb_to_see;
                                }
                            });
                            // Listen as long as you have to
                            while todo_seen.iter().any(|s| *s != 0) {
                                let c = connection.next_channel();
                                // TODO get expected unmapped next channel
                                running_counter = running_counter.wrapping_add(1);
                                assert_eq!(running_counter, connection.cur_event_counter);
                                let unmapped = csa2_no_subevent_unmapped(running_counter as u32, connection.channel_id as u32);
                                extra_events += 1;
                                total_nb_events += 1;
                                // Check if you actually heard it
                                if unmapped == c && rng.gen_range(0.0..1.0) <= 1.0 - packet_loss && todo_seen[unmapped as usize] != 0 {
                                    // Set to used 
                                    running_chm |= 1 << c;
                                    todo_seen[c as usize] = 0;
                                }
                                else {
                                    // Always remember you listened for it
                                    if todo_seen[unmapped as usize] != 0 {
                                        todo_seen[unmapped as usize] -= 1;
                                    }
                                }
                            }
                            CounterInterval::ExactlyOneSolution(first_packet_actual_counter, running_chm, 0)
                        }
                    }
                    else {
                        result.0
                    };

                    //if nb_used == 37 { println!("Completed sim {} of {} for {}: {:?} {}", ns, NUMBER_SIMS, nb_used, &result, extra_packets)};
                    //if let Ok(p) = pb.lock() { p.inc(1); p.set_message(format!("{}", nb_used)) }
                    if let CounterInterval::ExactlyOneSolution(first_packet_actual_counter, found_chm, _) = &r {
                        (((*first_packet_actual_counter == relative_event_offset && connection.channel_map == *found_chm), true), result.1, extra_packets, nb_used - nb_used_observed, total_nb_events, extra_events)
                    }
                    else {
                        ((false,false), result.1, extra_packets, nb_used - nb_used_observed, total_nb_events, extra_events)
                    }
                }).collect::<Vec<_>>();
                //todo.lock().unwrap().retain(|x| *x != nb_used);
                //println!("{:?} todo", todo.lock().unwrap());
                (nb_used, resses)
            }).collect::<Vec<_>>();

            let mut error_path = error_path.clone();
            error_path.push(format!("sim_err_{}bfs_{:.2}err_{:.}pl.png",  bfs_max, max_error, packet_loss));
            File::create(error_path.clone()).expect("Failed to create plot file");
            let mut bf_path = bf_path.clone();
            bf_path.push(format!("sim_bf_{}bfs_{:.2}err_{:.}pl.png",  bfs_max, max_error, packet_loss));
            File::create(bf_path.clone()).expect("Failed to create plot file");
            let mut extra_path = extra_path.clone();
            extra_path.push(format!("sim_extra_{}bfs_{:.2}err_{:.}pl.png",  bfs_max, max_error, packet_loss));
            File::create(extra_path.clone()).expect("Failed to create plot file");
            let mut nt_path = nt_path.clone();
            nt_path.push(format!("sim_nt_{}bfs_{:.2}err_{:.}pl.png",  bfs_max, max_error, packet_loss));
            File::create(nt_path.clone()).expect("Failed to create plot file");
            let mut extra_events_path = extra_events_path.clone();
            extra_events_path.push(format!("sim_extra_events_{}bfs_{:.2}err_{:.}pl.png",  bfs_max, max_error, packet_loss));
            File::create(extra_events_path.clone()).expect("Failed to create plot file");


            let mut seperate_fn_path = seperate_fn_path.clone();
            seperate_fn_path.push(format!("{}bfs_{:.2}err_{:.}pl",  bfs_max, max_error, packet_loss));
            create_dir_all(seperate_fn_path.clone()).expect("Failed to create plot directory");



            // return 3 iterators with the boxlots
            let mut successes = Vec::new();
            //let mut no_sols = Vec::new();
            let mut bfs  = Vec::new();
            let mut extras = Vec::new();
            let mut fns = Vec::new();
            let mut total_events = Vec::new();
            let mut extra_events = Vec::new();
            let mut seperate_fn_dist = Vec::new();
            for (nb_used, data) in sims {

                //println!("\n\n{} used", nb_used);
                

                let mut isuccesses = Vec::new();
                let mut ino_sols = Vec::new();
                let mut ibfs  = Vec::new();
                let mut iextra = Vec::new();
                let mut ifns = Vec::new();
                let mut itotal_events = Vec::new();
                let mut iextra_events = Vec::new();
                for d in data {
                    isuccesses.push(d.0.0);
                    ino_sols.push(!d.0.1);
                    ibfs.push(d.1);
                    iextra.push(d.2);
                    ifns.push(d.3);
                    itotal_events.push(d.4 as f64);
                    iextra_events.push(d.5 as f64);
                }

                //println!("{} success {} no sol", successes.iter().filter(|s| **s).count() as f64 / NUMBER_SIMS as f64, no_sols.iter().filter(|s| **s).count() as f64 / NUMBER_SIMS as f64);
                //println!("\n\nsuccesses\n{:?}\n\nno_sols\n{:?}\n\nbfs\n{:?}\n\nextra\n{:?}\n\nfns\n{:?}", &successes, &no_sols, &bfs, &extra, &fns);

                let successes_e = isuccesses.into_iter().filter(|b| *b).count() as f64 / NUMBER_SIMS as f64;
                let bfs_quarts = Quartiles::new(ibfs.as_slice());
                let extra_quarts = Quartiles::new(iextra.as_slice());
                let fns_quarts = Quartiles::new(&ifns);
                let nt_quarts = Quartiles::new(itotal_events.as_slice());
                let extra_event_quarts = Quartiles::new(iextra_events.as_slice());

                successes.push((nb_used, successes_e));
                //no_sols.push()
                bfs.push(Boxplot::new_vertical(SegmentValue::CenterOf(nb_used as u32), &bfs_quarts));
                extras.push(Boxplot::new_vertical(SegmentValue::CenterOf(nb_used as u32), &extra_quarts));
                fns.push(Boxplot::new_vertical(nb_used as u32, &fns_quarts));
                total_events.push(Boxplot::new_vertical(SegmentValue::CenterOf(nb_used as u32), &nt_quarts));
                extra_events.push(Boxplot::new_vertical(SegmentValue::CenterOf(nb_used as u32), &extra_event_quarts));
                seperate_fn_dist.push((nb_used, ifns));
            }

            const HEIGHT: u32 = 1080;
            const WIDTH: u32 = 1080; // was 1920

            // Successes/errors
            let root_area = BitMapBackend::new(error_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
            root_area.fill(&WHITE).unwrap();
            let mut events_chart = ChartBuilder::on(&root_area)
                .set_label_area_size(LabelAreaPosition::Left, 120)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .caption(format!("sim errors: {} max feasable brute force {:.2}% max error with {:.}% packet loss: {} events {:.2} thress", bfs_max, max_error, packet_loss, events, thress), ("sans-serif", 20))
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

            // BFS
            let root_area = BitMapBackend::new(bf_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
            root_area.fill(&WHITE).unwrap();
            let mut events_chart = ChartBuilder::on(&root_area)
                .set_label_area_size(LabelAreaPosition::Left, 120)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .caption(format!("sim brute forces: {} max feasable brute force {:.2}% max error with {:.}% packet loss: {} events {:.2} thress", bfs_max, max_error, packet_loss, events, thress), ("sans-serif", 20))
                .margin(20)
                .build_cartesian_2d((1..38u32).into_segmented(), 0.0..(bfs_max as f32 + 5.0))
                .expect("Chart building failed.");
            events_chart
                .configure_mesh()
                .disable_x_mesh()
                .y_desc("Brute forces")
                .x_desc("Actual used channels")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw()
                .unwrap();
            events_chart.draw_series(bfs).unwrap();


            // extras
            let root_area = BitMapBackend::new(extra_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
            root_area.fill(&WHITE).unwrap();
            let mut events_chart = ChartBuilder::on(&root_area)
                .set_label_area_size(LabelAreaPosition::Left, 120)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .caption(format!("sim extras: {} max feasable brute force {:.2}% max error with {:.}% packet loss: {} events {:.2} thress", bfs_max, max_error, packet_loss, events, thress), ("sans-serif", 20))
                .margin(20)
                .build_cartesian_2d((1..38u32).into_segmented(), 0.0..20.0f32)
                .expect("Chart building failed.");
            events_chart
                .configure_mesh()
                .disable_x_mesh()
                .y_desc("Extra packets after channel map phase")
                .x_desc("Actual used channels")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw()
                .unwrap();
            events_chart.draw_series(extras).unwrap();



            // total events
            let root_area = BitMapBackend::new(nt_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
            root_area.fill(&WHITE).unwrap();
            let mut events_chart = ChartBuilder::on(&root_area)
                .set_label_area_size(LabelAreaPosition::Left, 120)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .caption(format!("sim total events: {} max feasable brute force {:.2}% max error with {:.}% packet loss: {} events {:.2} thress", bfs_max, max_error, packet_loss, events, thress), ("sans-serif", 20))
                .margin(20)
                .build_cartesian_2d((1..38u32).into_segmented(), 0.0..10000.0f32)
                .expect("Chart building failed.");
            events_chart
                .configure_mesh()
                .disable_x_mesh()
                .y_desc("Total #events")
                .x_desc("Actual used channels")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw()
                .unwrap();
            events_chart.draw_series(total_events).unwrap();



            // extra events
            let root_area = BitMapBackend::new(extra_events_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
            root_area.fill(&WHITE).unwrap();
            let mut events_chart = ChartBuilder::on(&root_area)
                .set_label_area_size(LabelAreaPosition::Left, 120)
                .set_label_area_size(LabelAreaPosition::Bottom, 60)
                .caption(format!("sim extra events: {} max feasable brute force {:.2}% max error with {:.}% packet loss: {} events {:.2} thress", bfs_max, max_error, packet_loss, events, thress), ("sans-serif", 20))
                .margin(20)
                .build_cartesian_2d((1..38u32).into_segmented(), 0.0..1000.0f32)
                .expect("Chart building failed.");
            events_chart
                .configure_mesh()
                .disable_x_mesh()
                .y_desc("Extra events for checking TODO")
                .x_desc("Actual used channels")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw()
                .unwrap();
            events_chart.draw_series(extra_events).unwrap();


            for (nb_used, fns) in seperate_fn_dist {
                let mut seperate_fn_path = seperate_fn_path.clone();
                seperate_fn_path.push(format!("{}_used.png",  nb_used));
                File::create(seperate_fn_path.clone()).expect("Failed to create plot file");

                let root_area = BitMapBackend::new(seperate_fn_path.as_path(), (WIDTH, HEIGHT)).into_drawing_area();
                root_area.fill(&WHITE).unwrap();


                let mut events_chart = ChartBuilder::on(&root_area)
                    .set_label_area_size(LabelAreaPosition::Left, 120)
                    .set_label_area_size(LabelAreaPosition::Bottom, 60)
                    .caption(format!("FNS: {} used {} events {} thresshold {} sims", nb_used, events, thress, fns.len()), ("sans-serif", 20))
                    .margin(20)
                    .build_cartesian_2d(0..38u32, 0.0..1.05f64)
                    .expect("Chart building failed.");
                events_chart
                    .configure_mesh()
                    .disable_x_mesh()
                    .y_desc("P(#False negatives | #Used)")
                    .x_desc("#False negatives")
                    .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                    .axis_desc_style(("sans-serif", 20))
                    .draw()
                    .unwrap();

                // Draw theoretical
                let txt =  |coord : (u32, f64), size : i32, style: ShapeStyle| {
                    EmptyElement::at(coord) 
                    + Circle::new((0, 0), size, style)   
                    + Text::new(if coord.1 > thress {format!("{:.2}", coord.1)} else {"".to_string()}, (10, -10), ("sans-serif", 10))
                };
                let t = PointSeries::of_element(
                    (0..=nb_used).map(|nb_false_neg| (nb_false_neg as u32, chance_and_combo_reality(37 -nb_used + nb_false_neg, nb_false_neg,events, 1.0 - packet_loss).0)),
                    5,
                    RED.filled(),
                    &txt,
                );
                events_chart.draw_series(t).unwrap()
                .label("Theoretical")
                .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], RED.filled()));
                
                // Draw observed
                let l = fns.len();
                let c = LineSeries::new(
                    (0..=nb_used).map(|nb_false_neg| (nb_false_neg as u32, fns.iter().filter(|f| **f == nb_false_neg).count() as f64 / l as f64 )), BLUE.stroke_width(3));
                events_chart.draw_series(c).unwrap()
                .label("Observed")
                .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], BLUE.filled()));


                let t = LineSeries::new(
                    vec![(0, thress), (37, thress)].into_iter(),
                     BLACK.stroke_width(3));
                events_chart.draw_series(t).unwrap()
                .label("Thresshold")
                .legend(move |(x, y)| Rectangle::new([(x, y + 7), (x + 14, y - 7)], BLACK.filled()));
            }


        }
    });
}

fn geo_qdf(p: f64, wanted_probability : f64) -> u32 {
    let raw = (1.0f64 - wanted_probability).log(1.0 - p);
    //println!("{}", raw);
    raw.ceil() as u32 
}

#[allow(clippy::too_many_arguments)]
fn brute_force(_extra_packets: u32 ,_bf_max: u64,actual_nb_used_debug :u8, actual_chm_debug : u64, actual_counter_debug : u16, packets : &[(u16, u8)], chm : u64, thresshold: f64, nb_events: u8, packet_loss: f64, channel_id: u16) -> (CounterInterval, u32) {
    //if actual_nb_used_debug == 37 { println!("bf for {} {} {}", actual_nb_used_debug, actual_counter_debug, actual_chm_debug)};
    let nb_unused_seen = (0u8..37).filter(|channel| chm & (1 << *channel) == 0).count() as u8;
    // Get the false positives for which the chance of it occurring is above the thresshold
    let likely_false_negatives = (0..=nb_unused_seen)
        .filter(|fns| chance_and_combo_reality(nb_unused_seen, *fns,nb_events, 1.0 - packet_loss).0 >= thresshold).collect_vec();
    if likely_false_negatives.is_empty() {
        //if actual_nb_used_debug < 37 {
        //    let t = (0..=nb_unused_seen)
        //    .map(|fns| chance_and_combo_reality(nb_unused_seen, fns,nb_events, 1.0 - packet_loss).0 ).collect_vec();
        //    t.iter().enumerate().for_each(|(l,d)| println!("{} {:.2}", *d, l));
        //    println!("above thress = {:.2}", thresshold);
        //    std::io::stdout().flush();
        //    panic!("");
        //}
        return (CounterInterval::NoSolutions,0);
    }
        // TODO turn as much as you can of this into iterators so rust can optimise the hell out of it

    let (channel_map_bool_array,_, _, _) =
        generate_channel_map_arrays(chm);
    let unused_channels = channel_map_bool_array.iter().enumerate().filter_map(|(channel, seen)| if !*seen {Some(channel as u8)} else {None}).collect_vec();
    let mut nb_bfs = 0u32;
    let result = likely_false_negatives.into_iter().map(|nb_false_negs| {
        
        //let nb_used = 37 - nb_unused_seen + nb_false_negs;
        //let mut is_false_neg = unused_channels.iter().map(|_| false).collect_vec();
        //is_false_neg.iter_mut().zip(0..nb_false_negs).for_each(|(is_false_neg, _)| *is_false_neg = true);
        //let nb_unused = nb_unused_seen - nb_false_negs;
        // permutation takes K elements from the iterator and gives a vector for each combination of k element of the iterator
        // Taking nb_unused_seen - false_nges is same as deleting false_negs
        let combinations = unused_channels.clone().into_iter().combinations(nb_false_negs as usize).collect_vec();
        let chms = combinations.clone().into_iter().map(|to_flip| {
        unused_channels.iter().fold(0x1F_FF_FF_FF_FFu64, |chm, channel|{
                if !to_flip.contains(channel) {
                    // turn of if now flipped to used
                    chm & !(1 << *channel)
                }
                else {
                    chm
                }
            })
        }).collect_vec();
        let _nb_u = actual_nb_used_debug;
        let b = binomial(nb_unused_seen as u64, nb_false_negs as u64) as usize;
        if b != chms.len() {
            panic!("{} {:?}", b, combinations)
        }
        //let fn_solutions = 
        chms.into_iter().map( |chm|{
            let (channel_map_bool_array,remapping_table, _, nb_used) =  generate_channel_map_arrays(chm);

            //nb_bfs += 1;
            //if nb_bfs > bf_max as u32 {
            //    panic!("More bfs than allowed")
            //}
            // now we have concrete channel map as before
            let mut running_event_counter;

            let mut found_counter: Option<u32> = None;
            let mut inconsistency: bool;
            for potential_counter in 0..=u16::MAX {
                // reset inconsistency
                inconsistency = false;
                for (relative_counter, channel) in packets.iter() {
                    running_event_counter =  potential_counter.wrapping_add(*relative_counter);
                    let channel_potential_counter = csa2_no_subevent(
                        running_event_counter as u32,
                        channel_id as u32,
                        &channel_map_bool_array,
                        &remapping_table,
                        nb_used,
                    );

                    // If we get another one than expected, go to next counter
                    if channel_potential_counter != *channel {
                        inconsistency = true;
                        if potential_counter == actual_counter_debug && chm == actual_chm_debug {
                            panic!("Correct counter and channel map but inconsistency")
                        }
                        break;
                    }
                }


                // If there was no inconsistency for this potential counter save it
                if !inconsistency {
                    match found_counter {
                        None => {
                            // the first one without inconsistency, save it
                            found_counter = Some(potential_counter as u32);
                        }
                        Some(_) => {
                            // There was already another one without inconstistency, we have multiple solutions
                            return CounterInterval::MultipleSolutions;
                        }
                    }
                }
            }

            // The fact we get here, we did not find mutliple solutions, must be one or none.
            // Remember for exactly one you need to run through the whole range
            match found_counter {
                None => {
                    // There were no solutions
                    if chm == actual_chm_debug {
                        panic!("No solution but have actual channel map")
                    }
                    CounterInterval::NoSolutions
                }
                Some(counter) => 
                    CounterInterval::ExactlyOneSolution(counter as u16, chm, 0),
            }
        })
        //.collect_vec();
        //(nb_false_negs, fn_solutions)
    })//.collect_vec();
    .flatten().inspect(|_| nb_bfs += 1).reduce(|a,b| { // IMPORTANT if 1 element this will give the 1 element => one solution stays one solution
        match a {
            CounterInterval::ExactlyOneSolution(ac, am, atodo) => {
                match b {
                    CounterInterval::ExactlyOneSolution(bc, bm, btodo) => {
                        // now require them to be exactly the same -> Wrong e.g. 0 fns wont have this
                        //assert!((*m).count_ones() <= tm.count_ones());
                        if ac == bc  {
                            // For same counters, or them = take union of used channels
                            // Later on we will require the union to be exactly the same for all numbers of false positives
                            // Remembers chm + the places where a 1 was turned into a 0
                            let more_todo_from_both = am ^ bm;

                            CounterInterval::ExactlyOneSolution(ac, am & bm,atodo | btodo | more_todo_from_both)
                        }
                        else {
                            CounterInterval::MultipleSolutions
                        }
                    }
                    CounterInterval::MultipleSolutions => {CounterInterval::MultipleSolutions}
                    CounterInterval::NoSolutions => {a}
                }
            }
            CounterInterval::MultipleSolutions => {CounterInterval::MultipleSolutions}
            CounterInterval::NoSolutions => {b}
        }
    }).unwrap();

    (result, nb_bfs)
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CounterInterval {
    /// Acceptable end state if it is the only one.
    ExactlyOneSolution(u16, u64, u64),
    /// Indicates there were mutliple solutions and we need more information
    MultipleSolutions,
    /// If no solution for any slice error. Otherwise ok.
    NoSolutions,
}
