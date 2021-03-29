use rand::prelude::*;
use std::fmt;

struct Node {
    x: f32,
    y: f32,
}

impl Node {
    pub fn new(x: f32, y: f32) -> Node {
        Node { x, y }
    }
    pub fn rand_new(min: f32, max: f32) -> Node {
        let mut rng = thread_rng();
        let x: f32 = rng.gen_range(min, max).floor();
        let y: f32 = rng.gen_range(min, max).floor();
        Node { x, y }
    }
    pub fn distance_to(&self, node: &Node) -> f32 {
        (((self.x - node.x) * (self.x - node.x)) + ((self.y - node.y) * (self.y - node.y))).sqrt()
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    for _ in 1..10 * 1000 {
        println!("{}", Node::rand_new(-20.0, 20.0));
    }
    let node_1 = Node::new(1.0, 0.0);
    let node_2 = Node::new(0.0, 2.0);
    println!(
        "{} distance_to {}, {}",
        node_1,
        node_2,
        node_1.distance_to(&node_2)
    );
}
