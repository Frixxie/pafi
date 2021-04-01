use rand::prelude::*;
use std::fmt;
extern crate sdl2;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;

use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Node {
    x: f32,
    y: f32,
}

impl Node {
    pub fn new(x: f32, y: f32) -> Node {
        Node { x, y }
    }

    pub fn rand_new(min_x: f32, max_x: f32, min_y: f32, max_y: f32) -> Node {
        let mut rng = thread_rng();
        let x: f32 = rng.gen_range(min_x, max_x).floor();
        let y: f32 = rng.gen_range(min_y, max_y).floor();
        Node { x, y }
    }

    pub fn distance_to(&self, node: &Node) -> f32 {
        (((self.x - node.x) * (self.x - node.x)) + ((self.y - node.y) * (self.y - node.y))).sqrt()
    }

    pub fn into_point(&self) -> sdl2::rect::Point {
        sdl2::rect::Point::new(self.x as i32, self.y as i32)
    }

    pub fn into_rect(&self, w: i32, h: i32) -> sdl2::rect::Rect {
        sdl2::rect::Rect::new(
            (self.x as i32) - w / 2,
            (self.y as i32) - h / 2,
            w as u32,
            h as u32,
        )
    }

    pub fn create_rand_nodes(
        n: usize,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Vec<Node> {
        (0..n)
            .into_par_iter()
            .map(|_| Node::rand_new(min_x, max_x, min_y, max_y))
            .collect()
    }

    pub fn calc_path(nodes: &Vec<Node>) -> f32 {
        nodes
            .par_iter()
            .enumerate()
            .map(|(i, node)| node.distance_to(&nodes[(i + 1) % (nodes.len())]))
            .sum()
    }

    /// Traveling salesman problem next nearest neighbour
    pub fn tsp_nnn(nodes: Vec<Node>) -> Vec<Node> {
        let reses: Vec<Vec<Node>> = (0..nodes.len())
            .into_par_iter()
            .map(|i| {
                let mut cpy = nodes.to_vec();
                let mut reses_inner: Vec<Node> = Vec::new();
                let mut j: usize = i % nodes.len();
                while cpy.len() > 0 {
                    let node = cpy.remove(j);
                    reses_inner.push(node);
                    let mut min: f32 = std::f32::INFINITY;
                    let mut min_k: usize = 0;
                    for (k, other_node) in cpy.iter().enumerate() {
                        let tmp = node.distance_to(other_node);
                        if tmp < min {
                            min = tmp;
                            min_k = k;
                        }
                    }
                    j = min_k
                }
                // println!("{}, {}", i, Node::calc_path(&reses_inner));
                reses_inner
            })
            .collect();
        let paths_len: Vec<f32> = reses
            .par_iter()
            .map(|nodes| Node::calc_path(nodes))
            .collect();

        let mut min: f32 = std::f32::INFINITY;
        let mut min_i: usize = 0;
        for (i, len) in paths_len.iter().enumerate() {
            // println!("{}, {}, {}, {}", i, len, min, min_i);
            if len < &min {
                min = *len;
                min_i = i;
            }
        }
        println!(
            "{}, {}, {}",
            min,
            min_i,
            reses
                .par_iter()
                .map(|nodes| Node::calc_path(nodes))
                .sum::<f32>()
                / paths_len.len() as f32
        );
        reses[min_i].to_vec()
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    //demo of using distance_to
    let node_1 = Node::new(400.0, 300.0);
    let node_2 = Node::new(0.0, 2.0);
    println!(
        "{} distance_to {}, {}",
        node_1,
        node_2,
        node_1.distance_to(&node_2)
    );

    //setting up and solving rand nodes
    let nodes_unord = Node::create_rand_nodes(2048, 10.0, 1590.0, 10.0, 990.0);
    let nodes = Node::tsp_nnn(nodes_unord);

    //setting up sdl
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("PaFi", 1600, 1000)
        .position_centered()
        .build()
        .unwrap();
    let mut canvas = window.into_canvas().build().unwrap();
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();

        //keyevents
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        //drawing nodes
        for (i, node) in nodes.iter().enumerate() {
            canvas.set_draw_color(Color::RGB(255, 100, 55));
            canvas.draw_rect(node.into_rect(8, 8)).unwrap();
            canvas.set_draw_color(Color::RGB(255, 255, 255));
            if i < (nodes.len() - 1) {
                canvas
                    .draw_line(node.into_point(), nodes[i + 1].into_point())
                    .unwrap();
            } else {
                canvas
                    .draw_line(node.into_point(), nodes[0].into_point())
                    .unwrap();
            }
        }
        canvas.present();
    }
}
