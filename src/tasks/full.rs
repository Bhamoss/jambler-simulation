use itertools::Itertools;
use itertools::max;
use jambler::ble_algorithms::crc;
use jambler::ble_algorithms::csa2::calculate_channel_identifier;
use jambler::ble_algorithms::whitening;
use jambler::jambler::deduction::brute_force::brute_force;
use jambler::jambler::deduction::control::*;
use jambler::jambler::deduction::deducer::*;
use plotters::prelude::*;
use rand::Rng;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::{fs::{create_dir_all, File}};


use std::f64;

use crate::csa2::BlePhy;
use crate::{SimulationParameters, tasks::BleConnection};


use std::sync::{Arc, Mutex};
use indicatif::{MultiProgress};


// IMPORTANT does not take into account the possible 16 and distance drift. This is much too restrictive and cripples this approach
// IMPORTANT from my understanding this is software time. Mention this in thesis
pub fn full<R: RngCore + Send + Sync>(mut params: SimulationParameters<R>, _bars: Arc<Mutex<MultiProgress>>) {
    params.output_dir.push("full");
    create_dir_all(&params.output_dir).unwrap();
    
    
    let file_path = params.output_dir;
    let mut rng = params.rng;
    // Dir for per nb unused observed
    //file_path.push("capture_chance_sim");
    //create_dir_all(file_path.clone()).expect("Failed to create plot directory");

    // First one gets taken as the number of events to plot them all together
    let start_params = vec![
        DeductionStartParameters {
            access_address: 0, // NA
            master_phy: BlePhy::Uncoded1M, 
            slave_phy: BlePhy::CodedS2, 
            packet_loss: 0.2,
            nb_sniffers: 10,
            conn_interval_success_rate: 0.95,
            channel_map_success_rate: 0.9,
            anchor_point_success_rate: 0.96,
            silence_percentage: 0.1,
            max_brute_forces: 100,
        }, 
        DeductionStartParameters {
            access_address: 0, // NA
            master_phy: BlePhy::Uncoded1M, 
            slave_phy: BlePhy::Uncoded2M, 
            packet_loss: 0.2,
            nb_sniffers: 10,
            conn_interval_success_rate: 0.9,
            channel_map_success_rate: 0.9,
            anchor_point_success_rate: 0.95,
            silence_percentage: 0.3,
            max_brute_forces: 100,
        },
        DeductionStartParameters {
            access_address: 0, // NA
            master_phy: BlePhy::Uncoded2M, 
            slave_phy: BlePhy::Uncoded2M, 
            packet_loss: 0.4,
            nb_sniffers: 5,
            conn_interval_success_rate: 0.9,
            channel_map_success_rate: 0.9,
            anchor_point_success_rate: 0.95,
            silence_percentage: 0.1,
            max_brute_forces: 100,
        },
        DeductionStartParameters {
            access_address: 0, // NA
            master_phy: BlePhy::Uncoded2M, 
            slave_phy: BlePhy::Uncoded2M, 
            packet_loss: 0.4,
            nb_sniffers: 5,
            conn_interval_success_rate: 0.9,
            channel_map_success_rate: 0.9,
            anchor_point_success_rate: 0.95,
            silence_percentage: 0.1,
            max_brute_forces: 1000,
        },
        DeductionStartParameters {
            access_address: 0, // NA
            master_phy: BlePhy::CodedS8, 
            slave_phy: BlePhy::CodedS8, 
            packet_loss: 0.5,
            nb_sniffers: 5,
            conn_interval_success_rate: 0.9,
            channel_map_success_rate: 0.9,
            anchor_point_success_rate: 0.95,
            silence_percentage: 0.1,
            max_brute_forces: 100,
        } 
    ];

    struct Ret {
        success: bool,
        time : u64,
        conn_ok: bool,
        crc_ok: bool,
        counter_ok : bool,
        chm_ok : bool,
    }
    
    let sims = start_params.into_iter().map(|n| (n, ChaCha20Rng::seed_from_u64(rng.next_u64())))
    .collect_vec();
    // TODO into par iter
    sims.into_par_iter().enumerate().for_each(|(enu,(start_params, mut urng))| {

        
        let mut file_path = file_path.clone();
        file_path.push(format!("full_{:.2}_pl_{}_sniffs_{}_bfs_{}",start_params.packet_loss, start_params.nb_sniffers, start_params.max_brute_forces, enu));
        create_dir_all(file_path.clone()).expect("Failed to create plot directory");


        let sims = (7u8..=37).step_by(5).map(|n| (n, ChaCha20Rng::seed_from_u64(urng.next_u64())))
        .collect_vec();

        const NUMBER_SIMS : u32 = 100;


        // TODO into par iter
        sims.into_par_iter().for_each(|(nb_used, mut rng)| {
            // PLOT PER NUMBER USED
            const STEP : u32 = 100; // gewoon priem
            // (7500u32..=4000000).step_by((1250 * STEP) as usize) vec![7500, 50000, 400000, 1000000, 4000000]
            let conn_int_sims = vec![7500, 50000, 125000, 400000, 1000000, 2000000, 4000000].into_iter().map(|conn_interval| {
                // TODO die par bridge .par_bridge()
                let sims = (0..NUMBER_SIMS).map(|s| (s, ChaCha20Rng::seed_from_u64(rng.next_u64()))).par_bridge().map(|(_sim, mut rng)|{

                    println!("{} used {} conn {} sim", nb_used, conn_interval, _sim);

                    // gen random connection
                    let mut connection = BleConnection::new(&mut rng, Some(nb_used as u8));
                    connection.connection_interval = conn_interval;
                    // Enforce the ppm max
                    /*
                    if connection.connection_interval as f64 * connection.master_ppm as f64 / 1_000_000.0 > 624.5 {
                        let legal_ppm_max = (625.0 * 1_000_000.0 / connection.connection_interval as f64) as u16;
                        if !(156..500).contains(&legal_ppm_max) {
                            panic!("ubvak")
                        }
                        connection.master_ppm = rng.gen_range(10..=legal_ppm_max);
                    }
                    */
                    let pos_drift = connection.master_ppm as f64 / 1_000_000.0;
                    let max_subevent_time = subevent_time(255,  &start_params.master_phy) + 152 + subevent_time(255, &start_params.slave_phy) + 152; 
                    let shortest_pos_event_time = (conn_interval as f64 * (1.0 - connection.master_ppm as f64 / 1000000.0)) as u64;

                    if max_subevent_time > shortest_pos_event_time {
                        return Ret {
                            success: true,
                            time: 0,
                            crc_ok: false,
                            conn_ok: false,
                            counter_ok: false,
                            chm_ok: false,
                        };
                    }

                    // Startparams need to get the random access address
                    let mut start_params = start_params.clone();
                    start_params.access_address = connection.access_address;

                    // Capture chance * correct crc = 1.0 - packet loss;
                    let phy_capture_chance = (1.0 - start_params.packet_loss as f64).sqrt();
                    let crc_correct_chance = phy_capture_chance;

                    struct ActualAP {
                        pub channel: u8, 
                        pub time: u64, 
                        pub event: u16,
                    }

                    let mut aps = vec![];

                    struct Sniffer {
                        channel: u8, 
                        time_last_channel_noise: u64,
                        start_time_channel: u64,
                        id : u8
                    }
                    // Create start state
                    let mut running_time = 0;
                    let mut sniffers = (0..start_params.nb_sniffers).map(|c| Sniffer{ channel: c, time_last_channel_noise: running_time, start_time_channel:running_time, id: c}).collect_vec();
                    static mut DPBUF : DpBuf = MaybeUninit::uninit();
                    static mut BFPBUF : BfpBuf = MaybeUninit::uninit();
                    let mut store = DeductionQueueStore::new();
                    let (mut control,mut state) = unsafe{store.split(&mut DPBUF, &mut BFPBUF)};

                    // Start jambler
                    control.start(start_params);
                    state.deduction_loop();
                    let time_to_switch = if let DeducerToMaster::SearchPacketsForCrcInit(time_to_switch) = control.get_deducer_request().unwrap() {
                        time_to_switch
                    } else {panic!("")};

                    let mut start_time_sniffers_on_channel = running_time;

                    let silence_time_ded = (time_to_switch as f32 * start_params.silence_percentage).ceil() as u32;

                    // Get packets in random order until further instructions
                    let mut found = state.deduction_loop().0;
                    while !found {
                        let event_start_time = running_time;
                        let mut packs_this_event = 0;
                        // Send packets in event as long as they fit
                        while (event_start_time + shortest_pos_event_time) - running_time > max_subevent_time && packs_this_event < 10{
                            packs_this_event += 1;

                            // Check if heard. Packet loss is worst case here, namely
                            let new_samples = sniffers.iter_mut().filter_map(|s| {
                                if s.channel == connection.cur_channel && rng.gen_range(0.0..1.0) <= phy_capture_chance {
                                    let response = if rng.gen_range(0.0..1.0) <= phy_capture_chance {Some(ConnectionSamplePacket{
                                        first_header_byte: 0,
                                        reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                        phy: start_params.slave_phy,
                                        rssi: 0,
                                    })} else {None};
                                    let r = ConnectionSample{
                                        slave_id: s.id,
                                        channel: s.channel,
                                        time: running_time,
                                        silence_time_on_channel: (running_time - s.time_last_channel_noise) as u32,
                                        packet: ConnectionSamplePacket{
                                            first_header_byte: 0,
                                            reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                            phy: start_params.master_phy,
                                            rssi: 0,
                                        },
                                        response,
                                    };
                                    s.time_last_channel_noise = running_time;
                                    if silence_time_ded < r.silence_time_on_channel && r.packet.reversed_crc_init == connection.crc_init {
                                        aps.push(ActualAP{
                                            channel: connection.cur_channel,
                                            time: running_time,
                                            event: connection.cur_event_counter,
                                        })
                                    }
                                    Some(r)
                                } else {None}
                            }).next();

                            // Check if we found a sample. Stop if jambler ask us so.
                            if let Some(sample) = new_samples {
                                control.send_connection_sample(sample);
                                if state.deduction_loop().0 {found = true;break}
                            }


                            // Generate the time the next packets would have been sent
                            let master_packet_time = subevent_time(rng.gen_range(0..=255), &start_params.master_phy);
                            running_time += master_packet_time + rng.gen_range(0..=4);
                            let slave_packet_time = subevent_time(rng.gen_range(0..=255), &start_params.master_phy);
                            running_time += slave_packet_time + rng.gen_range(0..=4);

                            // Let jamblers jump
                            while running_time >= start_time_sniffers_on_channel + time_to_switch as u64 {
                                start_time_sniffers_on_channel += time_to_switch as u64;
                                sniffers.iter_mut().for_each(|s| {
                                    s.channel = (s.channel + start_params.nb_sniffers) % 37; 
                                    s.start_time_channel = start_time_sniffers_on_channel;
                                    s.time_last_channel_noise = s.start_time_channel;
                                });
                            }


                        }

                        // Event done, go to next
                        connection.next_channel();
                        running_time = event_start_time + (conn_interval as f64 * (1.0 - rng.gen_range(-pos_drift..pos_drift))) as u64;
                        // Let jamblers jump
                        while running_time >= start_time_sniffers_on_channel + time_to_switch as u64 {
                            start_time_sniffers_on_channel += time_to_switch as u64;
                            sniffers.iter_mut().for_each(|s| {
                                s.channel = (s.channel + start_params.nb_sniffers) % 37; 
                                s.start_time_channel = start_time_sniffers_on_channel;
                                s.time_last_channel_noise = s.start_time_channel;
                            });
                        }
                    }
                    
                    // Should have crc init right now
                    let (time_to_switch, crc_init, used_channels_until_now)  = if let DeducerToMaster::SearchPacketsForConnInterval(t,s,v) = control.get_deducer_request().unwrap() {(t,s,v)} else {panic!("")};

                    let crc_ok = crc_init == connection.crc_init && state.get_crc_init() == crc_init ;

                    if !crc_ok {
                        return Ret {
                            success: false,
                            time: running_time,
                            crc_ok,
                            conn_ok: false,
                            counter_ok: false,
                            chm_ok: false,
                        };
                    }


                    // Keep doing the same, but let sniffers stay on channels when they captured one on it
                    let mut known_used_channels = [false;37];
                    (0u8..37).filter(|c| used_channels_until_now & (1 << *c) != 0).for_each(|c| known_used_channels[c as usize] = true);
                    let mut sniffers = (0..start_params.nb_sniffers).map(|c| Sniffer{ channel: c, time_last_channel_noise: running_time, start_time_channel:running_time, id: c}).collect_vec();
                    // Put as many sniffers on used seen channel as you can
                    known_used_channels.iter().enumerate().filter_map(|(idx, b)| if *b {Some(idx as u8)} else {None}).zip(sniffers.iter_mut()).for_each(|(c,s)| s.channel = c);

                    for i in (known_used_channels.iter().filter(|d| **d).count())..sniffers.len() {
                        // Do not have to check for unused but known used channels, the finder will stay on it
                        // Stays on same channel if no free ones
                        for pos_next_channel in (1..37).map(|l| (sniffers[i].channel + l) % 37) {
                            if (0..sniffers.len()).all(|j| sniffers[j].channel != pos_next_channel) {
                                sniffers[i].channel = pos_next_channel;
                                break
                            }
                        }
                    }
                    /* 
                    for i in 0..sniffers.len() {
                        for j in 0..sniffers.len() {
                            if i != j && sniffers[i].channel == sniffers[j].channel && sniffers.len() <= 37 {
                                panic!("")
                            }
                        }
                    }
                    */

                    let mut found = state.deduction_loop().0;
                    while !found {
                        let event_start_time = running_time;
                        let mut packs_this_event = 0;
                        // Send packets in event as long as they fit
                        while  (event_start_time + shortest_pos_event_time) - running_time > max_subevent_time && packs_this_event < 10{
                            packs_this_event += 1;

                            // Check if heard. Packet loss is worst case here, namely
                            let new_samples = sniffers.iter_mut().filter_map(|s| {
                                if s.channel == connection.cur_channel && rng.gen_range(0.0..1.0) <= phy_capture_chance {
                                    let response = if rng.gen_range(0.0..1.0) <= phy_capture_chance {Some(ConnectionSamplePacket{
                                        first_header_byte: 0,
                                        reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                        phy: start_params.slave_phy,
                                        rssi: 0,
                                    })} else {None};
                                    let r = ConnectionSample{
                                        slave_id: s.id,
                                        channel: s.channel,
                                        time: running_time,
                                        silence_time_on_channel: (running_time - s.time_last_channel_noise) as u32,
                                        packet: ConnectionSamplePacket{
                                            first_header_byte: 0,
                                            reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                            phy: start_params.master_phy,
                                            rssi: 0,
                                        },
                                        response,
                                    };
                                    s.time_last_channel_noise = running_time;
                                    if silence_time_ded < r.silence_time_on_channel && r.packet.reversed_crc_init == connection.crc_init {
                                        aps.push(ActualAP{
                                            channel: connection.cur_channel,
                                            time: running_time,
                                            event: connection.cur_event_counter,
                                        })
                                    }
                                     if aps.as_slice().windows(2).any(|w| {
                                            let d = (w[1].time - w[0].time) as f64; 
                                            let dr = ((d / conn_interval as f64).round()) * conn_interval as f64;
                                            (d - dr) > d * 500.0/1000000.0 } ) {
                                         //panic!("")
                                     }
                                    Some(r)
                                } else {None}
                            }).next();

                            // Check if we found a sample. Stop if jambler ask us so.
                            if let Some(sample) = new_samples {
                                known_used_channels[sample.channel as usize] = true;
                                control.send_connection_sample(sample);
                                if state.deduction_loop().0 {found = true;break}
                            }
                            // Generate the time the next packets would have been sent
                            let master_packet_time = subevent_time(rng.gen_range(0..=255), &start_params.master_phy);
                            running_time += master_packet_time + rng.gen_range(0..=4);
                            let slave_packet_time = subevent_time(rng.gen_range(0..=255), &start_params.master_phy);
                            running_time += slave_packet_time + rng.gen_range(0..=4);

                            // Let jamblers jump
                            while running_time >= start_time_sniffers_on_channel + time_to_switch as u64 {
                                start_time_sniffers_on_channel += time_to_switch as u64;
                                for i in 0..sniffers.len() {
                                    // Only do something to sniffers not on a used channel
                                    if !known_used_channels[sniffers[i].channel as usize] {
                                        // Do not have to check for unused but known used channels, the finder will stay on it
                                        // Stays on same channel if no free ones
                                        for pos_next_channel in (1..37).map(|l| (sniffers[i].channel + l) % 37) {
                                            if (0..sniffers.len()).all(|j| sniffers[j].channel != pos_next_channel) {
                                                sniffers[i].channel = pos_next_channel;
                                                break
                                            }
                                        }
                                        sniffers[i].start_time_channel = start_time_sniffers_on_channel;
                                        sniffers[i].time_last_channel_noise = start_time_sniffers_on_channel;
                                    }; 
                                }
                            }
                        }
                        // Event done, go to next
                        connection.next_channel();
                        running_time = event_start_time + (conn_interval as f64 * (1.0 - rng.gen_range(-pos_drift..pos_drift))) as u64;

                        // Let jamblers jump
                        while running_time >= start_time_sniffers_on_channel + time_to_switch as u64 {
                            start_time_sniffers_on_channel += time_to_switch as u64;
                            for i in 0..sniffers.len() {
                                // Only do something to sniffers not on a used channel
                                if !known_used_channels[sniffers[i].channel as usize] {
                                    // Do not have to check for unused but known used channels, the finder will stay on it
                                    // Stays on same channel if no free ones
                                    for pos_next_channel in (1..37).map(|l| (sniffers[i].channel + l) % 37) {
                                        if (0..sniffers.len()).all(|j| sniffers[j].channel != pos_next_channel) {
                                            sniffers[i].channel = pos_next_channel;
                                            break
                                        }
                                    }
                                    sniffers[i].start_time_channel = start_time_sniffers_on_channel;
                                    sniffers[i].time_last_channel_noise = start_time_sniffers_on_channel;
                                }; 
                            }
                        }
                    }
                    //println!("sim {} conn_int {}", _sim, conn_interval);
                    //return (true, running_time);

                    // Should have crc init right now
                    let (time_to_listen_in_us, channels_todo, crc_init)   = if let DeducerToMaster::StartChannelMap(t,s,v) = control.get_deducer_request().unwrap() {(t,s,v)} else {panic!("")};

                    let con = state.get_connection_interval();
                    let conn_ok = con == connection.connection_interval && state.get_connection_interval() == conn_interval ;

                    if con == 5000 {
                        print!("let packets : Vec<(u64, u8)> = vec![");
                        for a in aps.iter() {
                            print!("({}, {}) ,", a.time, a.channel);
                        }
                        println!("];");
                        /*
                        704990 - 599992
104998 -> inderdaad 2 events apart maar 5000 meer, vanwaar?
                        */
                    }

                    if !conn_ok {
                        return Ret {
                            success: false,
                            time: running_time,
                            crc_ok,
                            conn_ok,
                            counter_ok: false,
                            chm_ok: false,
                        };
                    }

                    if state.get_connection_interval() != conn_interval {panic!("")}

                    
                    let mut channels_still_todo = [false;37];
                    (0u8..37).filter(|c| channels_todo & (1 << *c) != 0).for_each(|c| channels_still_todo[c as usize] = true);

                    let tot_todo = channels_still_todo.iter().filter(|c| **c).count();

                    let mut sniffers = (0..start_params.nb_sniffers).map(|c| Sniffer{ channel: c, time_last_channel_noise: running_time, start_time_channel:running_time, id: c}).collect_vec();
                    // Put as many sniffers on todo seen channel as you can
                    channels_still_todo.iter_mut().enumerate().filter(|(_,b)| **b).zip(sniffers.iter_mut())
                    .for_each(|((c, b),s)| {*b = false; s.channel = c as u8});


                    for i in 0..sniffers.len() {
                        for j in 0..sniffers.len() {
                            if i != j && sniffers[i].channel == sniffers[j].channel && sniffers.len() <= tot_todo {
                                panic!("")
                            }
                        }
                    }

                    // take worst case, only 1 on channel
                    let mut found = state.deduction_loop().0;
                    while !found {
                        let event_start_time = running_time;
                        // Send packets in event as long as they fit

                        // Check if heard. Packet loss is worst case here, namely
                        let new_samples = sniffers.iter_mut().filter_map(|s| {
                            if s.channel == connection.cur_channel && rng.gen_range(0.0..1.0) <= phy_capture_chance {
                                let response = if rng.gen_range(0.0..1.0) <= phy_capture_chance {Some(ConnectionSamplePacket{
                                    first_header_byte: 0,
                                    reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                    phy: start_params.slave_phy,
                                    rssi: 0,
                                })} else {None};
                                let r = ConnectionSample{
                                    slave_id: s.id,
                                    channel: s.channel,
                                    time: running_time,
                                    silence_time_on_channel: (running_time - s.time_last_channel_noise) as u32,
                                    packet: ConnectionSamplePacket{
                                        first_header_byte: 0,
                                        reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                        phy: start_params.master_phy,
                                        rssi: 0,
                                    },
                                    response,
                                };
                                s.time_last_channel_noise = running_time;
                                if silence_time_ded < r.silence_time_on_channel && r.packet.reversed_crc_init == connection.crc_init {
                                    aps.push(ActualAP{
                                        channel: connection.cur_channel,
                                        time: running_time,
                                        event: connection.cur_event_counter,
                                    })
                                }
                                Some(r)
                            } else {None}
                        }).next();

                        // Check if we found a sample. Stop if jambler ask us so.
                        if let Some(sample) = new_samples {
                            // Only jump if you know jambler will accept (only set to used when crc is ok)
                            if sample.packet.reversed_crc_init == crc_init {
                                let snif = &mut sniffers[sample.slave_id as usize];
                                let new_channel = channels_still_todo.iter().enumerate().find(|(_,b)| **b)
                                .unwrap_or((snif.channel as usize, &false)).0 as u8;
                                snif.channel = new_channel;
                                channels_still_todo[new_channel as usize] = false;
                                snif.start_time_channel = running_time;
                                snif.time_last_channel_noise = running_time;
                            }
                            control.send_connection_sample(sample);
                            if state.deduction_loop().0 {found = true}
                        }



                        // Event done, go to next
                        connection.next_channel();
                        running_time = event_start_time + (conn_interval as f64 * (1.0 - rng.gen_range(-pos_drift..pos_drift))) as u64;

                        // Check jamblers for unused 
                        //IMPORTANT: time to listen here, not switch
                        sniffers.iter_mut().for_each(|snif|{
                            if running_time >= snif.start_time_channel + time_to_listen_in_us as u64 {
                                control.send_unused_channel(UnusedChannel{channel: snif.channel, sniffer_id: snif.id});
                                if state.deduction_loop().0 {found = true}
                                let new_channel = channels_still_todo.iter().enumerate().find(|(_,b)| **b)
                                .unwrap_or((snif.channel as usize, &false)).0 as u8;
                                snif.channel = new_channel;
                                channels_still_todo[new_channel as usize] = false;
                                snif.start_time_channel = running_time;
                                snif.time_last_channel_noise = running_time;
                            }
                        });

                        for i in 0..sniffers.len() {
                            for j in 0..sniffers.len() {
                                if i != j && sniffers[i].channel == sniffers[j].channel && sniffers.len() <= tot_todo {
                                    panic!("")
                                }
                            }
                        }
                    }

                    // Brute force
                    let (bfparams, channels_used)   = if let DeducerToMaster::DistributedBruteForce(t,s) = control.get_deducer_request().unwrap() {(t,s)} else {panic!("")};
                    let sols = (0..bfparams.nb_sniffers).map(|s| brute_force(s, bfparams.clone())).collect_vec();
                    //println!("{:?}", &sols);
                    sols.into_iter().for_each(|s| {control.send_brute_force_result(s); state.deduction_loop();});

                    let mut sniffers = (0..start_params.nb_sniffers).map(|c| Sniffer{ channel: c, time_last_channel_noise: running_time, start_time_channel:running_time, id: c}).collect_vec();

                    // Jambler can ask to immediately start processing again
                    while state.deduction_loop().1 {}

                    // Put as many sniffers on used channel as you can
                    (0u8..37).filter(|c| channels_used & (1 << c) != 0).zip(sniffers.iter_mut())
                    .for_each(|(c,s)| s.channel = c);

                    let mut req = control.get_deducer_request().unwrap();

                    while let DeducerToMaster::DistributedBruteForce(bfparams, _) = &req {
                        // Get another packet
                        let mut found = state.deduction_loop().0;
                        while !found {
                            let event_start_time = running_time;
                            // Send packets in event as long as they fit
                            // Generate the time the next packets would have been sent
                            let master_packet_time = subevent_time(rng.gen_range(0..=255), &start_params.master_phy);
                            running_time += master_packet_time + rng.gen_range(0..=4);
                            let slave_packet_time = subevent_time(rng.gen_range(0..=255), &start_params.master_phy);
                            running_time += slave_packet_time + rng.gen_range(0..=4);

                            // Check if heard. Packet loss is worst case here, namely
                            let new_samples = sniffers.iter_mut().filter_map(|s| {
                                if s.channel == connection.cur_channel && rng.gen_range(0.0..1.0) <= phy_capture_chance {
                                    let response = if rng.gen_range(0.0..1.0) <= phy_capture_chance {Some(ConnectionSamplePacket{
                                        first_header_byte: 0,
                                        reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                        phy: start_params.slave_phy,
                                        rssi: 0,
                                    })} else {None};
                                    let r = ConnectionSample{
                                        slave_id: s.id,
                                        channel: s.channel,
                                        time: running_time,
                                        silence_time_on_channel: (running_time - s.time_last_channel_noise) as u32,
                                        packet: ConnectionSamplePacket{
                                            first_header_byte: 0,
                                            reversed_crc_init: if rng.gen_range(0.0..1.0) <= crc_correct_chance {connection.crc_init} else {rng.next_u32()},
                                            phy: start_params.master_phy,
                                            rssi: 0,
                                        },
                                        response,
                                    };
                                    s.time_last_channel_noise = running_time;
                                    if silence_time_ded < r.silence_time_on_channel && r.packet.reversed_crc_init == connection.crc_init {
                                        aps.push(ActualAP{
                                            channel: connection.cur_channel,
                                            time: running_time,
                                            event: connection.cur_event_counter,
                                        })
                                    }
                                    Some(r)
                                } else {None}
                            }).next();

                            // Check if we found a sample. Stop if jambler ask us so.
                            if let Some(sample) = new_samples {
                                control.send_connection_sample(sample);
                                if state.deduction_loop().0 {found = true}
                            }
                            // Event done, go to next
                            connection.next_channel();
                            running_time = event_start_time + (conn_interval as f64 * (1.0 - rng.gen_range(-pos_drift..pos_drift))) as u64;

                        }

                        let sols = (0..bfparams.nb_sniffers).map(|s| brute_force(s, bfparams.clone())).collect_vec();
                        sols.into_iter().for_each(|s| {control.send_brute_force_result(s); state.deduction_loop();});
                        req = control.get_deducer_request().unwrap();
                    };


                    let (unsure_params, chs_todo) = match req {
                        DeducerToMaster::SearchPacketsForCrcInit(_) => {return Ret {
                            success: false,
                            time: running_time,
                            conn_ok,
                            crc_ok,
                            counter_ok: false,
                            chm_ok: false,
                        }}
                        DeducerToMaster::ListenForUnsureChannels(par, un) => {(par, un)}
                        _ => {panic!("")}
                    };


                    let mut next_counter = unsure_params.last_counter.wrapping_add(((running_time - unsure_params.last_time) as f32 / unsure_params.conn_interval as f32).round()  as u16 + 1 );
                    let channel_id = calculate_channel_identifier(unsure_params.access_address);

                    let was_correct_next = next_counter == connection.cur_event_counter + 1;


                    let correct_chm_still_possible = connection.channel_map & !(unsure_params.channel_map | chs_todo) == 0;


                    // Jambler can ask to immediately start processing again
                    while state.deduction_loop().1 {}

                    // IMPORTANT what if chs_todo = 0?

                    let mut req = control.get_deducer_request();
                    while req.is_none() {
                        loop {
                            running_time += (conn_interval as f64 * (1.0 - rng.gen_range(-pos_drift..pos_drift))) as u64;
                            let expected_next_unmapped_channel = jambler::ble_algorithms::csa2::csa2_unmapped(next_counter, channel_id);
                            let c = connection.next_channel();
                            next_counter = next_counter.wrapping_add(1);
                            // chm is 0 for unsure so unmapped
                            if chs_todo & (1 << expected_next_unmapped_channel) != 0 {
                                if c == expected_next_unmapped_channel && rng.gen_range(0.0..1.0) <= 1.0 - start_params.packet_loss {
                                    control.send_unsure_channel_event(UnsureChannelEvent{
                                        channel: expected_next_unmapped_channel,
                                        time: running_time,
                                        event_counter: next_counter,
                                        seen: true,
                                    })
                                }
                                else {
                                    control.send_unsure_channel_event(UnsureChannelEvent{
                                        channel: expected_next_unmapped_channel,
                                        time: running_time,
                                        event_counter: next_counter,
                                        seen: false,
                                    })
                                }
                                break
                            }
                        }
                        state.deduction_loop();
                        req = control.get_deducer_request();
                    }
                    
                    let found_par = match req.unwrap() {
                        DeducerToMaster::SearchPacketsForCrcInit(_) => {return Ret {
                            success: false,
                            time: running_time,
                            conn_ok,
                            crc_ok,
                            counter_ok: was_correct_next,
                            chm_ok: correct_chm_still_possible,
                        }}
                        DeducerToMaster::DeducedParameters(par) => {par}
                        _ => {panic!("")}
                    };

                    let expected_par = DeducedParameters {
                        access_address: start_params.access_address,
                        master_phy: start_params.master_phy,
                        slave_phy: start_params.slave_phy,
                        conn_interval: connection.connection_interval,
                        channel_map: connection.channel_map,
                        crc_init: connection.crc_init,
                        last_time: found_par.last_time,
                        last_counter: found_par.last_counter,
                    };
                    Ret {
                        success: found_par ==  expected_par,
                        time: running_time,
                        conn_ok,
                        crc_ok,
                        counter_ok: was_correct_next,
                        chm_ok: correct_chm_still_possible,
                    }
                }).collect::<Vec<_>>();
                (conn_interval, sims)
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
                .caption(format!("Params: {} used, {:.2} packet loss, {} sniffers, {:.2} ap, {:.2} silence, {} {} phys", 
                    nb_used, start_params.packet_loss, start_params.nb_sniffers, start_params.anchor_point_success_rate, start_params.silence_percentage
                    , phy_to_string_short(&start_params.master_phy), phy_to_string_short(&start_params.slave_phy)), 
                    ("sans-serif", 20))
                .margin(20)
                .right_y_label_area_size(80)
                .build_cartesian_2d(plotters::prelude::IntoLinspace::step(7500u32..4000001, 1250 * STEP), 0.0..1.05f64)
                .expect("Chart building failed.")
                .set_secondary_coord(plotters::prelude::IntoLinspace::step(7500u32..4000001, 1250 * STEP).into_segmented(), 0.0..2000.0f32);
            events_chart
                .configure_mesh()
                .disable_x_mesh()
                .y_desc("Success rate")
                .x_desc("Connection interval")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw()
                .unwrap();

            events_chart.configure_secondary_axes()
                .y_desc("Total time in seconds")
                .label_style(("sans-serif", 20)) // The style of the numbers on an axis
                .axis_desc_style(("sans-serif", 20))
                .draw().unwrap();

            let mut successes = vec![];
            let mut times = vec![];
            let mut conn_ok = vec![];
            let mut crc_ok = vec![];
            let mut counter_ok = vec![];
            let mut chm_ok = vec![];
            for (conn_int, sims) in conn_int_sims {
                let mut c_successes = vec![];
                let mut c_times = vec![];
                let mut c_conn_ok = vec![];
                let mut c_crc_ok = vec![];
                let mut c_counter_ok = vec![];
                let mut c_chm_ok = vec![];
                for r in sims {
                    c_successes.push(r.success);
                    c_crc_ok.push(r.crc_ok);
                    c_conn_ok.push(r.conn_ok);
                    c_counter_ok.push(r.counter_ok);
                    c_chm_ok.push(r.chm_ok);

                    if r.success {
                        c_times.push(r.time)
                    }
                }

                println!("{} used con {} t {:.}",nb_used, conn_int, c_times.iter().sum::<u64>() as f64 / c_times.len() as f64);

                let rate = c_successes.iter().filter(|v| **v).count() as f64 / c_successes.len() as f64;
                successes.push((conn_int, rate));

                let rate = c_crc_ok.iter().filter(|v| **v).count() as f64 / c_successes.len() as f64;
                crc_ok.push((conn_int, rate));

                let rate = c_conn_ok.iter().filter(|v| **v).count() as f64 / c_successes.len() as f64;
                conn_ok.push((conn_int, rate));

                let rate = c_counter_ok.iter().filter(|v| **v).count() as f64 / c_successes.len() as f64;
                counter_ok.push((conn_int, rate));

                let rate = c_chm_ok.iter().filter(|v| **v).count() as f64 / c_successes.len() as f64;
                chm_ok.push((conn_int, rate));

                if !c_times.is_empty(){
                    let c_times = c_times.into_iter().map(|t| t as f32 / 1000000.0).collect_vec();
                    let boxe = Boxplot::new_vertical(SegmentValue::Exact(conn_int), &Quartiles::new(c_times.as_slice()));
                    times.push(boxe);
                }
            }
            
            //let o = LineSeries::new(
            //    probs.iter().map(|p| (p.0 as u32, p.4)),
            //    BLUE.stroke_width(3));
            events_chart.draw_secondary_series(times).unwrap()
            .label("Time in seconds")
            .legend(move |(x, y)| Circle::new((x, y), 4, BLACK.filled()));

            // Draw rate

            let color = Palette99::pick(4);
            let c = LineSeries::new(
                crc_ok.into_iter(), 
                color.stroke_width(3));
            events_chart.draw_series(c).unwrap()
            .label("Crc")
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

            

            let color = Palette99::pick(1);
            let c = LineSeries::new(
                conn_ok.into_iter(), 
                color.stroke_width(3));
            events_chart.draw_series(c).unwrap()
            .label("Connection interval")
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
            
            let color = Palette99::pick(2);
            let c = LineSeries::new(
                counter_ok.into_iter(), 
                color.stroke_width(3));
            events_chart.draw_series(c).unwrap()
            .label("Counter")
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

            let color = Palette99::pick(3);
            let c = LineSeries::new(
                chm_ok.into_iter(), 
                color.stroke_width(3));
            events_chart.draw_series(c).unwrap()
            .label("Chm")
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
            
            let c = LineSeries::new(
                successes.into_iter(), 
                RED.stroke_width(2));
            events_chart.draw_series(c).unwrap()
            .label("Success rate")
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

    println!("Full done");
}


fn phy_to_string_short(phy: &BlePhy) -> &str {
    match phy {
        BlePhy::Uncoded1M => {"1M"}
        BlePhy::Uncoded2M => {"2M"}
        BlePhy::CodedS2 => {"S2"}
        BlePhy::CodedS8 => {"S8"}
    }
}


fn subevent_time(len: u8, phy: &BlePhy) -> u64 {
    static UNCODED_1M_SEND_TIME: u64 = 8;
    static UNCODED_2M_SEND_TIME: u64 = 4;
    static CODED_S2_SEND_TIME: u64 = 16; // AA, CI, TERM1 in S8
    static CODED_S8_SEND_TIME: u64 = 64;
    match phy {
        // preamble, header, PDU, crc
        BlePhy::Uncoded1M => {(1 + 4 + 2 + len as u64  + 3) * UNCODED_1M_SEND_TIME}
        BlePhy::Uncoded2M => {(2 + 4 + 2 + len as u64  + 3) * UNCODED_2M_SEND_TIME}
        BlePhy::CodedS2 => {80 + 256 + 16 + 24 + (len as u64  + 3) * CODED_S2_SEND_TIME + 6}
        BlePhy::CodedS8 => {80 + 256 + 16 + 24 + (len as u64  + 3) * CODED_S8_SEND_TIME + 24}
    }
}