mod node;
use node::Node;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;

fn main() {
    const NUM_NODES: usize = 256;

    //setting up and solving rand nodes
    let nodes_unord = Node::create_rand_nodes(NUM_NODES, 10, 1590, 10, 990);
    let nodes_nnn = Node::tsp_nnn(&nodes_unord);
    let nodes_aco = Node::tsp_aco(&nodes_nnn, 8 * NUM_NODES);
    let nodes = Node::tsp_brute_search_optim(&nodes_aco);

    let intersections = Node::get_intersections(&nodes);
    println!("Intersections {}", intersections.len());

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

        // for node in intersections.iter() {
        //     canvas.set_draw_color(Color::RGB(255, 255, 255));
        //     canvas.draw_rect(node.into_rect(8, 8)).unwrap();
        // }

        //drawing nodes
        canvas.set_draw_color(Color::RGB(255, 100, 55));
        for (i, node) in nodes.iter().enumerate() {
            canvas
                .draw_line(node, nodes[(i + 1).rem_euclid(nodes.len())])
                .unwrap();
        }

        for (i, node) in nodes.iter().enumerate() {
            canvas.set_draw_color(Color::RGB(
                255,
                (100 + i.rem_euclid(255)) as u8,
                (55 + i.rem_euclid(255)) as u8,
            ));
            canvas.draw_rect(node.into_rect(8, 8)).unwrap();
        }

        canvas.present();
    }
}
