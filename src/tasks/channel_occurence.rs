use std::{fs::{File, create_dir_all}};
use itertools::Itertools;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use plotters::prelude::*;
use rand_chacha::ChaCha20Rng;
use jambler::ble_algorithms::{csa2::{generate_channel_map_arrays, calculate_channel_identifier, csa2_no_subevent}, access_address::is_valid_aa};
use stats::Frequencies;

pub fn channel_occurences<R>(root_path: &std::path::Path, rng: &mut R) where R: Rng {
    let mut dest_dir = root_path.to_path_buf();
    dest_dir.push("channel_occurrences");
    create_dir_all(&dest_dir).unwrap();
    events_until_occurrence(&dest_dir, rng);
}

fn events_until_occurrence<R: Rng>(dest_dir: &std::path::Path, rng: &mut R) {
    let mut file_path  = dest_dir.to_path_buf();
    file_path.push("events_until_occurrence.png");

    File::create(file_path.clone()).expect("Failed to create plot file");

    // Do a simulation
    let nb_unused = vec![0_u8, 5, 10, 20, 30];
    const NUMBER_SIM : u32 = 1000000;
    let sims = nb_unused.iter().map(|n| (*n, ChaCha20Rng::seed_from_u64(rng.next_u64()))).collect_vec();
    let sims = sims.into_par_iter().map(|(nb_unused, mut rng)| {
        let mut freqs = Frequencies::new();
        (0..NUMBER_SIM).into_iter().for_each(|_| {
            // Get random channels
            let mut unused_channels = 0;
            let mut channel_map : u64 = 0x1F_FF_FF_FF_FF;
            let mut mask;
            while unused_channels < nb_unused  {
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
            let mut counter = 0_u32;
            let chm_info = generate_channel_map_arrays(channel_map);
            let mut cur_channel = csa2_no_subevent(event_counter as u32, channel_id as u32, &chm_info.0, &chm_info.1, 37 - nb_unused);
            while channel != cur_channel {
                counter += 1;
                event_counter = event_counter.wrapping_add(1);
                cur_channel = csa2_no_subevent(event_counter as u32, channel_id as u32, &chm_info.0, &chm_info.1, 37 - nb_unused);
            }
            freqs.add(counter);
        });
        (nb_unused, freqs)
    }).collect::<Vec<_>>();


    const HEIGHT : u32 = 1080;
    const WIDTH : u32 = 1080; // was 1920
    // Get the brute pixel backend canvas
    let root_area = BitMapBackend::new(file_path.as_path(), (WIDTH, HEIGHT))
    .into_drawing_area();

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
    events_chart.configure_mesh()
        //.light_line_style(&WHITE)
        .disable_x_mesh()
        .y_desc("% not seen yet on {} samples.")
        .x_desc("Number of connection events before first channel occurrence.")
        //.set_all_tick_mark_size(20) -> the little line that come out of the graph
        .label_style(("sans-serif", 20)) // The style of the numbers on an axis
        .axis_desc_style(("sans-serif", 20))
        .draw()
        .unwrap();


    for (nb_unused, freqs) in sims.into_iter() {

        let color = Palette99::pick(nb_unused as usize).mix(0.5);

        let l = LineSeries::new(
            events_chart.x_range().map(|i| 
                (i, 1.0 - ((0..=i).map(|x|freqs.count(&x)).sum::<u64>() as f64 / NUMBER_SIM as f64))
            )
            , color.stroke_width(3));

        events_chart.draw_series(
            l).unwrap()
        .label(format!("{} unused", nb_unused))
        .legend(move |(x, y)| Rectangle::new([(x, y+ 7), (x + 14, y-7)], color.filled()));

        
    }


    // Draws the legend
    events_chart
    .configure_series_labels()
    .background_style(&WHITE.mix(0.8))
    .border_style(&BLACK)
    .label_font(("sans-serif", 15))
    .position(SeriesLabelPosition::UpperRight)
    .draw().unwrap();


}