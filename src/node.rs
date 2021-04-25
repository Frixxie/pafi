use float_cmp::ApproxEq;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use std::fmt;

#[derive(Clone, Copy, PartialEq)]
enum State {
    Visited,
    Unvisited,
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct Node {
    x: i32,
    y: i32,
}

impl From<Node> for sdl2::rect::Point {
    fn from(node: Node) -> sdl2::rect::Point {
        sdl2::rect::Point::new(node.x as i32, node.y as i32)
    }
}

impl From<&Node> for sdl2::rect::Point {
    fn from(node: &Node) -> sdl2::rect::Point {
        sdl2::rect::Point::new(node.x as i32, node.y as i32)
    }
}

impl From<Node> for (f32, f32) {
    fn from(node: Node) -> (f32, f32) {
        (node.x as f32, node.y as f32)
    }
}

impl From<&Node> for (f32, f32) {
    fn from(node: &Node) -> (f32, f32) {
        (node.x as f32, node.y as f32)
    }
}

impl Node {
    pub fn new(x: i32, y: i32) -> Node {
        Node { x, y }
    }

    pub fn rand_new(min_x: i32, max_x: i32, min_y: i32, max_y: i32) -> Node {
        let mut rng = thread_rng();
        let x: i32 = rng.gen_range(min_x, max_x);
        let y: i32 = rng.gen_range(min_y, max_y);
        Node { x, y }
    }

    pub fn distance_to(&self, node: &Node) -> f32 {
        let res = ((((self.x - node.x) * (self.x - node.x))
            + ((self.y - node.y) * (self.y - node.y))) as f32)
            .sqrt();
        if res > 0.0 {
            return res;
        }
        std::f32::MAX
    }

    /// See: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    /// For explaination on how this works
    pub fn line_intersect_point(&self, p2: Node, p3: Node, p4: Node) -> Option<(f32, f32)> {
        let d = ((self.x - p2.x) * (p3.y - p4.y)) - ((self.y - p2.y) * (p3.x - p4.x));
        if d == 0 {
            return None;
        }
        let t =
            ((self.x - p3.x) * (p3.y - p4.y) - (self.y - p3.y) * (p3.x - p4.x)) as f32 / d as f32;
        let u = ((p2.x - self.x) * (self.y - p3.y) - (p2.y - self.y) * (self.x - p3.x)) as f32
            / d as f32;
        if t >= 0.0 && t <= 1.0 {
            let x = self.x as f32 + t * (p2.x - self.x) as f32;
            let y = self.y as f32 + t * (p2.y - self.y) as f32;
            Some((x, y))
        } else if u >= 0.0 && u <= 1.0 {
            let x = p3.x as f32 + u * (p4.x - p3.x) as f32;
            let y = p3.y as f32 + u * (p4.y - p3.y) as f32;
            Some((x, y))
        } else {
            None
        }
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
        min_x: i32,
        max_x: i32,
        min_y: i32,
        max_y: i32,
    ) -> Vec<Node> {
        (0..n)
            .into_par_iter()
            .map(|_| Node::rand_new(min_x, max_x, min_y, max_y))
            .collect()
    }

    pub fn calc_path(nodes: &[Node], n: usize) -> f32 {
        nodes
            .par_iter()
            .enumerate()
            .map(|(i, node)| node.distance_to(&nodes[(i + 1).rem_euclid(n)]))
            .sum()
    }

    /// See answer: https://stackoverflow.com/questions/11907947/how-to-check-if-a-point-lies-on-a-line-between-2-other-points
    /// for explaination
    fn is_on_line(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> bool {
        // 0 -> x and 1 -> y
        let cross = ((p0.0 - p1.0) * (p2.1 - p1.1)) - ((p0.1 - p1.1) * (p2.0 - p1.0));
        if cross.approx_eq(0.0, (0.0, 2)) {
            return true;
        }
        false
    }

    pub fn get_intersections(nodes: &[Node]) -> Vec<Node> {
        //finds each intesecting point by using filter
        //(checks if there is an intersection)
        //and map (gets the point) instead of for loop and if statements
        //Gets all possible intersection
        let reses: Vec<(f32, f32)> = (0..nodes.len())
            .into_par_iter()
            .flat_map(|i| {
                (i + 2..nodes.len() + i)
                    .into_par_iter()
                    .filter_map(move |j| {
                        nodes[i].line_intersect_point(
                            nodes[(i + 1).rem_euclid(nodes.len())],
                            nodes[(j).rem_euclid(nodes.len())],
                            nodes[(j + 1).rem_euclid(nodes.len())],
                        )
                    })
            })
            .collect();

        let mut tmp = Vec::new();
        let mut cpy = reses.to_vec();
        while !cpy.is_empty() {
            let node = cpy.pop().unwrap();
            let mut instances = 0;
            for j in 0..nodes.len() {
                if Node::is_on_line(
                    node,
                    nodes[j].into(),
                    nodes[(j + 1).rem_euclid(nodes.len())].into(),
                ) {
                    instances += 1;
                }
            }
            if instances > 0 {
                tmp.push(node);
            }
        }
        println!("{} -> {}", reses.len(), tmp.len());
        let mut res: Vec<Node> = tmp
            .par_iter()
            .map(|node| Node::new(node.0.floor() as i32, node.1.floor() as i32))
            .collect();
        res.par_sort();
        res.dedup();
        res
    }

    fn choose_next_node(idx: usize, nodes: &mut [(State, Node)], weights: &mut [i32]) -> usize {
        let new_weights: Vec<f32> = (0..weights.len())
            .into_par_iter()
            .map(|i| {
                if nodes[i].0 == State::Unvisited {
                    weights[i] as f32 * (-nodes[idx].1.distance_to(&nodes[i].1) / 64.0).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let dist = WeightedIndex::new(&new_weights).unwrap();
        let mut rng = thread_rng();
        let mut index = dist.sample(&mut rng);
        //to prevent visiting self
        while idx == index {
            println!("{} is visited retrying", index);
            index = dist.sample(&mut rng);
        }
        nodes[index].0 = State::Visited;
        weights[index] += 1;
        index
    }

    //TODO: refactor and use better algorithm for this
    pub fn tsp_aco(nodes: &[Node]) -> Vec<Node> {
        let start_val = Node::calc_path(&nodes, nodes.len());
        let mut weights: Vec<Vec<i32>> = (0..nodes.len())
            .into_par_iter()
            .map(|_| vec![1; nodes.len()])
            .collect();
        let mut indexes: Vec<usize> = Vec::new();
        for _ in 0..2048 {
            let mut tmp: Vec<(State, Node)> = nodes
                .par_iter()
                .map(|node| (State::Unvisited, *node))
                .collect();
            let mut index = 0;
            let mut all_visited = false;
            tmp[0].0 = State::Visited;
            indexes = Vec::new();
            while !all_visited {
                index = Node::choose_next_node(index, &mut tmp, &mut weights[index]);
                indexes.push(index);
                all_visited = true;
                for node in &tmp {
                    if node.0 == State::Unvisited {
                        all_visited = false;
                        break;
                    }
                }
            }
        }
        let mut res: Vec<Node> = indexes.par_iter().map(|i| nodes[*i]).collect();
        res.insert(0, nodes[0]);
        println!("{} -> {}", start_val, Node::calc_path(&res, nodes.len()));
        res
    }

    /// Traveling salesman problem next nearest neighbour
    pub fn tsp_nnn(nodes: &[Node]) -> Vec<Node> {
        println!("BEFORE {}", Node::calc_path(&nodes, nodes.len()));
        let reses: Vec<Vec<Node>> = (0..nodes.len())
            .into_par_iter()
            .map(|i| {
                let mut cpy = nodes.to_vec();
                let mut reses_inner: Vec<Node> = Vec::new();
                let mut j: usize = i % nodes.len();
                while !cpy.is_empty() {
                    let node = cpy.remove(j);
                    reses_inner.push(node);
                    let mut min: f32 = std::f32::MAX;
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
                reses_inner
            })
            .collect();
        let paths_len: Vec<f32> = reses
            .par_iter()
            .map(|nodes| Node::calc_path(nodes, nodes.len()))
            .collect();

        let mut min: f32 = std::f32::MAX;
        let mut min_i: usize = 0;
        for (i, len) in paths_len.iter().enumerate() {
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
                .map(|nodes| Node::calc_path(nodes, nodes.len()))
                .sum::<f32>()
                / paths_len.len() as f32
        );
        println!(
            "AFTER: {}",
            Node::calc_path(&reses[min_i].to_vec(), reses[min_i].to_vec().len())
        );
        reses[min_i].to_vec()
    }

    /// Traveling salesman problem brute search optimalization
    /// Attemps to check if the current path has any better paths to use
    pub fn tsp_brute_search_optim(nodes: &[Node]) -> Vec<Node> {
        let mut cpy = nodes.to_vec();
        let len = cpy.len();
        let mut best_path = Node::calc_path(&cpy, len);
        let mut res = cpy.to_vec();
        let mut times_swapped = 1;
        while times_swapped > 0 {
            times_swapped = 0;
            for i in 0..cpy.len() {
                for j in i + 1..cpy.len() + 1 {
                    cpy.swap(i, j.rem_euclid(len));
                    let new_path = Node::calc_path(&cpy, len);
                    if new_path < best_path {
                        println!("{} -> {}, {}, {} <-> {}", best_path, new_path, best_path - new_path, i, j);
                        best_path = new_path;
                        res = cpy.to_vec();
                        times_swapped += 1;
                    } else {
                        cpy.swap(j.rem_euclid(len), i);
                    }
                }
            }
        }
        res
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
