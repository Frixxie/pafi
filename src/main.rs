use rand::prelude::*;
use std::fmt;
extern crate sdl2;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;

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
        let mut nodes: Vec<Node> = Vec::new();
        for _ in 0..n {
            nodes.push(Node::rand_new(min_x, max_x, min_y, max_y));
        }
        nodes
    }
    pub fn traveling_sailsman_greedy(nodes: Vec<Node>) -> Vec<Node> {
        let mut cpy = nodes.to_vec();
        let mut reses: Vec<Node> = Vec::new();
        let mut i: usize = 0;
        while (cpy.len()) > 0 {
            let node = cpy.remove(i);
            reses.push(node);
            let mut min: f32 = std::f32::INFINITY;
            let mut min_j: usize = 0;
            for (j, other_nodes) in cpy.iter().enumerate() {
                let tmp = node.distance_to(other_nodes);
                if tmp < min {
                    min = tmp;
                    min_j = j;
                }
            }
            i = min_j
        }
        reses
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let node_1 = Node::new(400.0, 300.0);
    let node_2 = Node::new(0.0, 2.0);
    println!(
        "{} distance_to {}, {}",
        node_1,
        node_2,
        node_1.distance_to(&node_2)
    );
    let nodes_unord = Node::create_rand_nodes(50, 10.0, 790.0, 10.0, 590.0);
    let nodes = Node::traveling_sailsman_greedy(nodes_unord);
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("PaFi", 800, 600)
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
        for (i, node) in nodes.iter().enumerate() {
            canvas.set_draw_color(Color::RGB(255 - i as u8, 255 - i as u8, 255 - i as u8));
            canvas.draw_rect(node.into_rect(10, 10)).unwrap();
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
