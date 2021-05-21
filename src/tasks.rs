
use jambler::ble_algorithms::{
    access_address::is_valid_aa,
    csa2::{calculate_channel_identifier, csa2_no_subevent, generate_channel_map_arrays},
};
use rand::{Rng, RngCore, SeedableRng};

use rand_chacha::ChaCha20Rng;

pub mod channel_occurrence;
pub mod channel_recovery;
pub mod conn_interval;

pub struct BleConnection {
    pub channel_map: u64,
    pub chm: [bool; 37],
    pub remapping_table: [u8; 37],
    pub nb_used: u8,
    pub channel_id: u16,
    pub access_address: u32,
    pub start_event_counter: u16,
    pub start_time: u64,
    pub connection_interval: u32,
    pub master_ppm: u16,
    pub crc_init: u32,
    pub cur_event_counter: u16,
    pub cur_channel: u8,
    pub cur_time: u64,
    rng: ChaCha20Rng,
}

impl BleConnection {
    pub fn new(rng: &mut impl RngCore, nb_used: Option<u8>) -> Self {
        // Channel map
        let given_nb_used = match nb_used {
            Some(n) => n,
            None => rng.gen_range(1u8..=37),
        };
        let nb_unused = 37 - given_nb_used;
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
        let (chm, remapping_table, _, nb_used) = generate_channel_map_arrays(channel_map);
        assert_eq!(given_nb_used, nb_used);

        // Access address
        let mut access_address = rng.next_u32();
        while !is_valid_aa(access_address, jambler::jambler::BlePhy::Uncoded1M) {
            access_address = rng.next_u32();
        }
        let channel_id = calculate_channel_identifier(access_address);

        // conn_interval
        let mut connection_interval = rng.gen_range(7_500..=4_000_000);
        connection_interval -= connection_interval % 1250;

        // Start params
        let start_event_counter = rng.next_u32() as u16;
        let start_time = rng.next_u64() & 0xFF_FF_FF_FF_FF_FF_FF; // Only keep 7 bytes to not time overflow

        // master ppm and crc init
        let master_ppm = rng.gen_range(10..=500);
        let crc_init = rng.next_u32() & 0xFF_FF_FF;

        // current
        let cur_event_counter = start_event_counter;
        let cur_time = start_time;
        let cur_channel = csa2_no_subevent(
            cur_event_counter as u32,
            channel_id as u32,
            &chm,
            &remapping_table,
            nb_used,
        );

        BleConnection {
            channel_map,
            chm,
            remapping_table,
            nb_used,
            channel_id,
            access_address,
            start_event_counter,
            start_time,
            connection_interval,
            master_ppm,
            crc_init,
            cur_event_counter,
            cur_channel,
            cur_time,
            rng: ChaCha20Rng::seed_from_u64(rng.next_u64()),
        }
    }

    pub fn next_channel(&mut self) -> u8 {
        self.cur_event_counter = self.cur_event_counter.wrapping_add(1);
        // const 16 + 3km range random
        //let mut new_time: f64 = (16 + 24) as f64 * self.rng.gen_range(0.0..1.0);
        let possible_drift_percentage = self.master_ppm as f64 / 1_000_000.0;
        let new_time = self.connection_interval as f64
            * (1.0
                + self
                    .rng
                    .gen_range(-possible_drift_percentage..possible_drift_percentage));
        self.cur_time += new_time.round() as u64;
        self.cur_channel = csa2_no_subevent(
            self.cur_event_counter as u32,
            self.channel_id as u32,
            &self.chm,
            &self.remapping_table,
            self.nb_used,
        );
        self.cur_channel
    }

}
