use itertools::Itertools;
use jambler::ble_algorithms::{
    access_address::is_valid_aa,
    csa2::{calculate_channel_identifier, csa2_no_subevent, generate_channel_map_arrays},
};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use stats::Frequencies;
use std::fs::{create_dir_all, File};

use crate::{run_tasks, SimulationParameters, Task};

pub fn channel_occurrences<R: RngCore + Send + Sync>(mut params: SimulationParameters<R>) {
    params.output_dir.push("channel_occurrences");
    create_dir_all(&params.output_dir).unwrap();
    let tasks: Vec<Box<dyn Task>> = vec![
        Box::new(events_until_occurrence),
        Box::new(events_until_occurrence_imperfect),
        Box::new(error_by_wait),
        Box::new(error_by_wait_imperfect),
    ];
    run_tasks(tasks, params);
}

fn events_until_occurrence<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    // TODO plot boxplots on this and theoretical as a couple of crosses which should fall on the graph

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("events_until_occurrence.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    // Do a simulation
    let nb_unused = vec![0_u8, 5, 10, 20, 30];
    const NUMBER_SIM: u32 = 100000;
    let sims = nb_unused
        .iter()
        .map(|n| (*n, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let sims = sims
        .into_par_iter()
        .map(|(nb_unused, mut rng)| {
            let mut freqs = Frequencies::new();
            (0..NUMBER_SIM).into_iter().for_each(|_| {
                // Get random channels
                let mut unused_channels = 0;
                let mut channel_map: u64 = 0x1F_FF_FF_FF_FF;
                let mut mask;
                while unused_channels < nb_unused {
                    mask = 1 << ((rng.next_u32() as u8) % 37);
                    if channel_map & mask != 0 {
                        channel_map ^= mask;
                        unused_channels += 1;
                    }
                }
                let mut channel = (rng.next_u32() as u8) % 37;
                while channel_map & (1 << channel) == 0 {
                    channel = (rng.next_u32() as u8) % 37;
                }
                let mut access_address = rng.next_u32();
                while !is_valid_aa(access_address, jambler::jambler::BlePhy::Uncoded1M) {
                    access_address = rng.next_u32();
                }
                let channel_id = calculate_channel_identifier(access_address);
                let mut event_counter = rng.next_u32() as u16;
                let mut counter = 1_u32;
                let chm_info = generate_channel_map_arrays(channel_map);
                let mut cur_channel = csa2_no_subevent(
                    event_counter as u32,
                    channel_id as u32,
                    &chm_info.0,
                    &chm_info.1,
                    37 - nb_unused,
                );
                while channel != cur_channel {
                    counter += 1;
                    event_counter = event_counter.wrapping_add(1);
                    cur_channel = csa2_no_subevent(
                        event_counter as u32,
                        channel_id as u32,
                        &chm_info.0,
                        &chm_info.1,
                        37 - nb_unused,
                    );
                }
                freqs.add(counter);
            });
            (nb_unused, freqs)
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
        .caption(format!("Distribution of number of events until first occurrence for {} samples per #unused channels", NUMBER_SIM), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(0..200_u32, 1), 0.0..1.0)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("% not seen yet.")
        .x_desc("Number of connection events before first channel occurrence.")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw theoretical
    for (nb_unused, _) in sims.iter() {
        let color = Palette99::pick(*nb_unused as usize + 37).to_rgba();

        // TODO geometrisch verdeelt ->  verander use statrs::distribution::Geometric; met cdf functie
        let theoretical_not_yet_seen =
            move |x: u32| (1.0 - (1.0 / (37 - *nb_unused) as f64)).powi(x as i32);

        //let u = *nb_unused;

        let l = PointSeries::of_element(
            events_chart
                .x_range()
                .step_by(10)
                .map(|i| (i, theoretical_not_yet_seen(i))),
            5,
            color.filled(),
            &{
                move |coord, size, style| {
                    let el = EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style);
                    let s = "".to_string();
                    //if *Box::new(u).deref() == 0 {
                    //s = "t".to_string();
                    //s = format!("{:.2}%", coord.1 * 100.0);
                    //}
                    let el = el + Text::new(s, (0, 15), ("sans-serif", 8));

                    el
                }
            },
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!("{} unused theoretical", nb_unused))
            .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
    }

    // Draw observed
    for (nb_unused, freqs) in sims.into_iter() {
        let color = Palette99::pick(nb_unused as usize).mix(0.5);

        let l = LineSeries::new(
            events_chart.x_range().map(|i| {
                (
                    i,
                    1.0 - ((0..=i).map(|x| freqs.count(&x)).sum::<u64>() as f64
                        / NUMBER_SIM as f64),
                )
            }),
            color.stroke_width(3),
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!("{} unused", nb_unused))
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

fn events_until_occurrence_imperfect<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    // TODO plot boxplots on this and theoretical as a couple of crosses which should fall on the graph

    let mut file_path = params.output_dir.clone();
    let mut rng = params.rng;
    file_path.push("events_until_occurrence_imperfect.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    // Do a simulation
    let nb_unused = vec![0_u8, 17];
    let capture_chances = vec![params.capture_chance, 1.0, 0.5];
    const NUMBER_SIM: u32 = 100000;
    let sims = nb_unused
        .iter()
        .cartesian_product(capture_chances.iter())
        .map(|(n, c)| (*n, *c, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let sims = sims
        .into_par_iter()
        .map(|(nb_unused, capture_chance, mut rng)| {
            let mut freqs = Frequencies::new();
            (0..NUMBER_SIM).into_iter().for_each(|_| {
                // Get random channels
                let mut unused_channels = 0;
                let mut channel_map: u64 = 0x1F_FF_FF_FF_FF;
                let mut mask;
                while unused_channels < nb_unused {
                    mask = 1 << ((rng.next_u32() as u8) % 37);
                    if channel_map & mask != 0 {
                        channel_map ^= mask;
                        unused_channels += 1;
                    }
                }
                let mut channel = (rng.next_u32() as u8) % 37;
                while channel_map & (1 << channel) == 0 {
                    channel = (rng.next_u32() as u8) % 37;
                }
                let mut access_address = rng.next_u32();
                while !is_valid_aa(access_address, jambler::jambler::BlePhy::Uncoded1M) {
                    access_address = rng.next_u32();
                }
                let channel_id = calculate_channel_identifier(access_address);
                let mut event_counter = rng.next_u32() as u16;
                let mut counter = 1_u32;
                let chm_info = generate_channel_map_arrays(channel_map);
                let mut cur_channel = csa2_no_subevent(
                    event_counter as u32,
                    channel_id as u32,
                    &chm_info.0,
                    &chm_info.1,
                    37 - nb_unused,
                );

                let mut captured = rng.gen_range(0.0..1.0) <= capture_chance;

                while !(channel == cur_channel && captured) {
                    counter += 1;
                    event_counter = event_counter.wrapping_add(1);
                    cur_channel = csa2_no_subevent(
                        event_counter as u32,
                        channel_id as u32,
                        &chm_info.0,
                        &chm_info.1,
                        37 - nb_unused,
                    );
                    captured = rng.gen_range(0.0..1.0) <= capture_chance;
                }
                freqs.add(counter);
            });
            (nb_unused, capture_chance, freqs)
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
        .caption(format!("Distribution of number of events until first occurrence for {} samples per #unused channels", NUMBER_SIM), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(0..200_u32, 1), 0.0..1.0)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("% not seen yet.")
        .x_desc("Number of connection events before first channel occurrence.")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw theoretical
    for (nb_unused, capture_chance, _) in sims.iter() {
        let color = if (*capture_chance - params.capture_chance).abs() < 0.0001 {
            Palette99::pick(*nb_unused as usize + 37).mix(1.0)
        } else {
            Palette99::pick(*nb_unused as usize + 37).mix(0.5)
        };

        let theoretical_not_yet_seen =
            move |x: u32| (1.0 - *capture_chance * (1.0 / (37 - *nb_unused) as f64)).powi(x as i32);

        //let u = *nb_unused;

        let l = PointSeries::of_element(
            events_chart
                .x_range()
                .step_by(10)
                .map(|i| (i, theoretical_not_yet_seen(i))),
            5,
            color.filled(),
            &{
                move |coord, size, style| {
                    let el = EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style);
                    let s = "".to_string();
                    //if *Box::new(u).deref() == 0 {
                    //s = "t".to_string();
                    //s = format!("{:.2}%", coord.1 * 100.0);
                    //}
                    let el = el + Text::new(s, (0, 15), ("sans-serif", 8));

                    el
                }
            },
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!("{} unused theoretical", nb_unused))
            .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
    }

    for (nb_unused, capture_chance, freqs) in sims.into_iter() {
        let color = if (capture_chance - params.capture_chance).abs() < 0.0001 {
            Palette99::pick(nb_unused as usize).mix(1.0)
        } else {
            Palette99::pick(nb_unused as usize).mix(0.5)
        };

        let l = LineSeries::new(
            events_chart.x_range().map(|i| {
                (
                    i,
                    1.0 - ((0..=i).map(|x| freqs.count(&x)).sum::<u64>() as f64
                        / NUMBER_SIM as f64),
                )
            }),
            color.stroke_width(3),
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!(
                "{} unused {:.2}%",
                nb_unused,
                (capture_chance * 10000.0).round() / 100.0
            ))
            .legend(move |(x, y)| Rectangle::new([(x, y), (x + 14, y)], color.filled()));
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

fn error_by_wait<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    // TODO plot boxplots on this and theoretical as a couple of crosses which should fall on the graph

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("error_by_wait.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    const TO_WAIT: u32 = 300;
    let wait_range = 0..(TO_WAIT + 1);

    // Do a simulation
    let nb_unused = vec![0_u8, 5, 10, 20, 30];
    const NUMBER_SIM: u32 = 10000;
    let sims = nb_unused
        .iter()
        .map(|n| (*n, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let sims = sims
        .into_par_iter()
        .map(|(nb_unused, mut rng)| {
            let mut freqs = Frequencies::new();
            // Simulate for every events to wait
            (0..NUMBER_SIM).for_each(|_| {
                // Get random connection slice
                let mut unused_channels = 0;
                let mut channel_map: u64 = 0x1F_FF_FF_FF_FF;
                let mut mask;
                let mut actual_channel_map_seen = [true; 37];
                while unused_channels < nb_unused {
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
                while !is_valid_aa(access_address, jambler::jambler::BlePhy::Uncoded1M) {
                    access_address = rng.next_u32();
                }
                let channel_id = calculate_channel_identifier(access_address);
                let mut event_counter = rng.next_u32() as u16;
                //let mut channel_map_seen = [false;37];
                let chm_info = generate_channel_map_arrays(channel_map);

                // Simulate for all channels in random order
                let mut channels = actual_channel_map_seen
                    .iter()
                    .enumerate()
                    .filter_map(|(channel, used)| if *used { Some(channel as u8) } else { None })
                    .collect_vec();
                channels.shuffle(&mut rng);
                let mut cur_channel = csa2_no_subevent(
                    event_counter as u32,
                    channel_id as u32,
                    &chm_info.0,
                    &chm_info.1,
                    37 - nb_unused,
                );

                // Too slow for large number of simulation -> reuse
                /*
                for channel in channels {
                    // Listen on a channel. Decide unused after events_to_wait events
                    let mut counter = 1_u32;
                    while channel != cur_channel && counter <= events_to_wait as u32 {
                        counter += 1;
                        event_counter = event_counter.wrapping_add(1);
                        cur_channel = csa2_no_subevent(event_counter as u32, channel_id as u32, &chm_info.0, &chm_info.1, 37 - nb_unused);
                    }
                    if channel == cur_channel {
                        channel_map_seen[channel as usize] = true;
                    }
                }

                let correct = channel_map_seen.iter().zip(actual_channel_map_seen.iter()).all(|(o, a)| *o == *a);

                // Remember if a mistake was made
                if !correct {
                    freqs.add(events_to_wait);
                }
                */
                let wrong_thress = channels
                    .into_iter()
                    .map(|channel| {
                        let mut counter = 1_u32;
                        for _ in wait_range.clone() {
                            if cur_channel == channel {
                                break;
                            }
                            counter += 1;
                            // eventcounter captured as mutable ref, will not all be same
                            event_counter = event_counter.wrapping_add(1);
                            cur_channel = csa2_no_subevent(
                                event_counter as u32,
                                channel_id as u32,
                                &chm_info.0,
                                &chm_info.1,
                                37 - nb_unused,
                            );
                        }
                        counter
                    })
                    .max()
                    .unwrap();

                // All less then max will have wrong channel
                for to_wait in 0..wrong_thress {
                    freqs.add(to_wait)
                }
            });
            (nb_unused, freqs)
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
        .caption(format!("Error rate wrong channel map given #events to wait, {} samples per #unused channels/events to wait", NUMBER_SIM), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(wait_range, 1), 0.0..1.0)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Error rate.")
        .x_desc("Number of connection events to classify channel as unused.")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw theoretical
    for (nb_unused, _) in sims.iter() {
        let color = Palette99::pick(*nb_unused as usize + 37).to_rgba();

        let theoretical_wrong_channel_map = move |x: u32| {
            let chance_wrong_per_channel = (1.0 - (1.0 / (37 - *nb_unused) as f64)).powi(x as i32);
            let chance_seen_per_channel = 1.0 - chance_wrong_per_channel;
            1.0 - chance_seen_per_channel.powi(37 - *nb_unused as i32)
        };

        //let u = *nb_unused;

        let l = PointSeries::of_element(
            events_chart
                .x_range()
                .step_by(2)
                .map(|i| (i, theoretical_wrong_channel_map(i))),
            2,
            color.filled(),
            &{
                move |coord, size, style| {
                    let el = EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style);
                    let s = "".to_string();
                    //if *Box::new(u).deref() == 0 {
                    //s = "t".to_string();
                    //s = format!("{:.2}%", coord.1 * 100.0);
                    //}
                    let el = el + Text::new(s, (0, 15), ("sans-serif", 8));

                    el
                }
            },
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!("{} unused theoretical", nb_unused))
            .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
    }

    // Draw observed
    for (nb_unused, freqs) in sims.into_iter() {
        let color = Palette99::pick(nb_unused as usize).mix(0.5);

        let l = LineSeries::new(
            events_chart
                .x_range()
                .map(|i| (i, freqs.count(&i) as f64 / NUMBER_SIM as f64)),
            color.stroke_width(3),
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!("{} unused", nb_unused))
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

fn error_by_wait_imperfect<R: RngCore + Send + Sync>(params: SimulationParameters<R>) {
    // TODO plot boxplots on this and theoretical as a couple of crosses which should fall on the graph

    let mut file_path = params.output_dir;
    let mut rng = params.rng;
    file_path.push("error_by_wait_imperfect.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    const TO_WAIT: u32 = 300;
    let wait_range = 0..(TO_WAIT + 1);

    // Do a simulation
    let nb_unused = vec![0_u8, 17];
    let capture_chances = vec![params.capture_chance, 1.0, 0.5];
    const NUMBER_SIM: u32 = 10000;
    let sims = nb_unused
        .iter()
        .cartesian_product(capture_chances.iter())
        .map(|(n, c)| (*n, *c, ChaCha20Rng::seed_from_u64(rng.next_u64())))
        .collect_vec();
    let sims = sims
        .into_par_iter()
        .map(|(nb_unused, capture_chance, mut rng)| {
            let mut freqs = Frequencies::new();
            // Simulate for every events to wait
            (0..NUMBER_SIM).for_each(|_| {
                // Get random connection slice
                let mut unused_channels = 0;
                let mut channel_map: u64 = 0x1F_FF_FF_FF_FF;
                let mut mask;
                let mut actual_channel_map_seen = [true; 37];
                while unused_channels < nb_unused {
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
                while !is_valid_aa(access_address, jambler::jambler::BlePhy::Uncoded1M) {
                    access_address = rng.next_u32();
                }
                let channel_id = calculate_channel_identifier(access_address);
                let mut event_counter = rng.next_u32() as u16;
                //let mut channel_map_seen = [false;37];
                let chm_info = generate_channel_map_arrays(channel_map);

                // Simulate for all channels in random order
                let mut channels = actual_channel_map_seen
                    .iter()
                    .enumerate()
                    .filter_map(|(channel, used)| if *used { Some(channel as u8) } else { None })
                    .collect_vec();
                channels.shuffle(&mut rng);
                let mut cur_channel = csa2_no_subevent(
                    event_counter as u32,
                    channel_id as u32,
                    &chm_info.0,
                    &chm_info.1,
                    37 - nb_unused,
                );

                let mut captured = rng.gen_range(0.0..1.0) <= capture_chance;

                let wrong_thress = channels
                    .into_iter()
                    .map(|channel| {
                        let mut counter = 1_u32;
                        for _ in wait_range.clone() {
                            if cur_channel == channel && captured {
                                break;
                            }
                            counter += 1;
                            // eventcounter captured as mutable ref, will not all be same
                            event_counter = event_counter.wrapping_add(1);
                            cur_channel = csa2_no_subevent(
                                event_counter as u32,
                                channel_id as u32,
                                &chm_info.0,
                                &chm_info.1,
                                37 - nb_unused,
                            );
                            captured = rng.gen_range(0.0..1.0) <= capture_chance;
                        }
                        counter
                    })
                    .max()
                    .unwrap();

                // All less then max will have wrong channel
                for to_wait in 0..wrong_thress {
                    freqs.add(to_wait)
                }
            });
            (nb_unused, capture_chance, freqs)
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
        .caption(format!("Error rate wrong channel map given #events to wait, {} samples per #unused channels/events to wait", NUMBER_SIM), ("sans-serif", 20))
        .margin(20)
        //.build_cartesian_2d(min..max, 0..100_000u32)
        .build_cartesian_2d(IntoLinspace::step(wait_range, 1), 0.0..1.0)
        //(infos.iter().map(|f| OrderedFloat::from(f.total_revenue)).max().unwrap().into_inner() * 1.1))
        .expect("Chart building failed.");

    // .format("%Y-%m-%d").to_string()

    // Do makeup
    events_chart
        .configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("Error rate.")
        .x_desc("Number of connection events to classify channel as unused.")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();

    // Draw theoretical
    for (nb_unused, capture_chance, _) in sims.iter() {
        let color = if (*capture_chance - params.capture_chance).abs() < 0.0001 {
            Palette99::pick(*nb_unused as usize + 37).mix(1.0)
        } else {
            Palette99::pick(*nb_unused as usize + 37).mix(0.5)
        };

        let theoretical_wrong_channel_map = move |x: u32| {
            let chance_seen_on_event = *capture_chance * (1.0 / (37 - *nb_unused) as f64);
            let chance_wrong_per_channel = (1.0 - chance_seen_on_event).powi(x as i32);
            let chance_seen_per_channel = 1.0 - chance_wrong_per_channel;
            1.0 - chance_seen_per_channel.powi(37 - *nb_unused as i32)
        };

        //let u = *nb_unused;

        let l = PointSeries::of_element(
            events_chart
                .x_range()
                .step_by(2)
                .map(|i| (i, theoretical_wrong_channel_map(i))),
            2,
            color.filled(),
            &{
                move |coord, size, style| {
                    let el = EmptyElement::at(coord) // have text and circle be relative to this
                    + Circle::new((0, 0), size, style);
                    let s = "".to_string();
                    //if *Box::new(u).deref() == 0 {
                    //s = "t".to_string();
                    //s = format!("{:.2}%", coord.1 * 100.0);
                    //}
                    let el = el + Text::new(s, (0, 15), ("sans-serif", 8));

                    el
                }
            },
        );

        events_chart.draw_series(l).unwrap();
        //.label(format!("{} unused theoretical", nb_unused))
        //.legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
    }

    // Draw observed
    for (nb_unused, capture_chance, freqs) in sims.into_iter() {
        let color = if (capture_chance - params.capture_chance).abs() < 0.0001 {
            Palette99::pick(nb_unused as usize + 37).mix(1.0)
        } else {
            Palette99::pick(nb_unused as usize + 37).mix(0.5)
        };

        let l = LineSeries::new(
            events_chart
                .x_range()
                .map(|i| (i, freqs.count(&i) as f64 / NUMBER_SIM as f64)),
            color.stroke_width(3),
        );

        events_chart
            .draw_series(l)
            .unwrap()
            .label(format!(
                "{} unused {:.2}%",
                nb_unused,
                (capture_chance * 10000.0).round() / 100.0
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
