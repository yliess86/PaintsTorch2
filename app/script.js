const IMG_W = 512;
const IMG_H = 512;
const SQR_IMG = IMG_W * IMG_H;

let draw = () => {};

function resize(scvs, sw, sh, nw, nh) {
    let cvs = document.createElement("canvas");
    let ctx = cvs.getContext("2d");
    cvs.height = nh;
    cvs.width = nw;

    ctx.drawImage(scvs, 0, 0, sw, sh, 0, 0, nw, nh);
    return ctx.getImageData(0, 0, nw, nh);
}


async function paintstorch(session, input, mask, hints, ctx) {    
    let input_from = ndarray(new Float32Array(input), [IMG_H, IMG_W, 4]);
    let hints_from = ndarray(new Float32Array(hints), [IMG_H / 4, IMG_W / 4, 4]);
    let mask_from  = ndarray(new Float32Array(mask),  [IMG_H, IMG_W, 4]);
    
    let input_to = ndarray(new Float32Array(4 * IMG_H * IMG_W), [1, 4, IMG_H, IMG_W]);
    let hints_to = ndarray(new Float32Array(4 * IMG_H / 4 * IMG_W / 4), [1, 4, IMG_H / 4, IMG_W / 4]);
    let mask_to = ndarray(new Float32Array(IMG_H * IMG_W), [1, IMG_H, IMG_W]);
    
    ndarray.ops.assign(input_from.pick(null, null, 3), mask_from.pick(null, null, 0));
    ndarray.ops.assign(mask_to.pick(0, null, null), mask_from.pick(null, null, 0));
    ndarray.ops.divseq(mask_to.pick(0, null, null), 255);

    for(let i = 0; i < 4; i++) {
        ndarray.ops.assign(input_to.pick(0, i, null, null), input_from.pick(null, null, i));
        ndarray.ops.assign(hints_to.pick(0, i, null, null), hints_from.pick(null, null, i));
        ndarray.ops.divseq(input_to.pick(0, i, null, null), 255);
        ndarray.ops.divseq(hints_to.pick(0, i, null, null), 255);
        
        if(i < 3) {
            ndarray.ops.subseq(input_to.pick(0, i, null, null), 0.5);
            ndarray.ops.subseq(hints_to.pick(0, i, null, null), 0.5);
            ndarray.ops.divseq(input_to.pick(0, i, null, null), 0.5);
            ndarray.ops.divseq(hints_to.pick(0, i, null, null), 0.5);
        }
    }
    
    let input_tensor = new Tensor(input_to.data, "float32", [1, 4, IMG_H, IMG_W]);
    let hints_tensor = new Tensor(hints_to.data, "float32", [1, 4, IMG_H / 4, IMG_W / 4]);
    let mask_tensor = new Tensor(mask_to.data, "float32", [1, IMG_H, IMG_W]);
    
    let output_map = await session.run([input_tensor, hints_tensor, mask_tensor]);
    let output_data = ndarray(new Float32Array(output_map.values().next().value.data), [1, 3, IMG_H, IMG_W]);
    let output_to = ndarray(new Float32Array(IMG_H * IMG_W * 4), [IMG_H, IMG_W, 4]);
    
    for(let i = 0; i < 3; i++) {
        ndarray.ops.mulseq(output_data.pick(0, i, null, null), 0.5);
        ndarray.ops.addseq(output_data.pick(0, i, null, null), 0.5);
        ndarray.ops.mulseq(output_data.pick(0, i, null, null), 255);
        ndarray.ops.assign(output_to.pick(null, null, i), output_data.pick(0, i, null, null));
    }

    ndarray.ops.assigns(output_to.pick(null, null, 3), 255);

    const data = Uint8ClampedArray.from(output_to.data);
    const img_data = new ImageData(data, IMG_W, IMG_H);
    ctx.putImageData(img_data, 0, 0);
}


const Tools = {
    PEN: "pen",
    ERASER: "eraser",
}


class Position {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}


class Brush {
    constructor(color, size, tool) {
        this.color = color;
        this.size = size;
        this.tool = tool;

        this.enabled = false;

        this.last_position = new Position(0, 0);
        this.current_position = new Position(0, 0);
    }

    draw_circle(ctx, color) {
        ctx.beginPath();
        ctx.fillStyle = null;
        ctx.strokeStyle = color;
        ctx.arc(
            this.current_position.x,
            this.current_position.y,
            this.size, 0, 2 * Math.PI
        );
        ctx.stroke();
        ctx.closePath();
    }

    enable() { this.enabled = true; }
    disable() { this.enabled = false; }
}


class DrawingCanvas {
    constructor(cvs, ctx, clear_color, cursor_color, color, size, tool) {
        this.cvs = cvs;
        this.ctx = ctx;
        this.clear_color = clear_color;
        this.cursor_color = cursor_color;

        this.data_cvs = document.createElement("canvas");
        this.data_ctx = this.data_cvs.getContext("2d");
        this.data_cvs.width = this.cvs.width;
        this.data_cvs.height = this.cvs.height;
        this.data_cvs.id = "data_" + this.cvs.id;

        this.default_brush = new Brush(color, size, tool);
        this.brush = new Brush(color, size, tool);

        this.clear();
        this.clear_data();

        this.cvs.addEventListener("mousemove", event => {
            this.brush.current_position = this.get_brush_position(event);
            
            this.clear();
            if (this.brush.enabled) this.draw();            
            this.ctx.drawImage(this.data_cvs, 0, 0);
            this.brush.draw_circle(this.ctx, this.cursor_color);
            
            this.brush.last_position = this.brush.current_position;
        });

        this.cvs.addEventListener("mousedown", event => { this.brush.enable(); });
        this.cvs.addEventListener("mouseup", event => { this.brush.disable(); });
    }

    reset() {
        this.brush.color = this.default_brush.color;
        this.brush.size = this.default_brush.size;
        this.brush.tool = this.default_brush.tool;
    }

    clear() {
        this.ctx.fillStyle = this.clear_color;
        this.ctx.fillRect(0, 0, this.cvs.width, this.cvs.height);
    }

    clear_data() {
        this.data_ctx.fillStyle = this.clear_color;
        this.data_ctx.fillRect(0, 0, this.data_cvs.width, this.data_cvs.height);
        this.clear();
        draw();
    }

    fill_data() {
        this.data_ctx.fillStyle = this.cursor_color;
        this.data_ctx.fillRect(0, 0, this.data_cvs.width, this.data_cvs.height);
        this.clear();
        draw();
    }

    get_brush_position(event) {
        let rect = this.cvs.getBoundingClientRect();
        return new Position(
            event.clientX - rect.left,
            event.clientY - rect.top
        );
    }

    draw() {
        this.data_ctx.beginPath();
        
        let is_pen = this.brush.tool == Tools.PEN;
        this.data_ctx.strokeStyle = is_pen? this.brush.color: this.clear_color;
        this.data_ctx.lineWidth = this.brush.size;
        
        this.data_ctx.moveTo(this.brush.last_position.x, this.brush.last_position.y);
        this.data_ctx.lineTo(this.brush.current_position.x, this.brush.current_position.y);
        this.data_ctx.lineJoin = this.data_ctx.lineCap = "round";
        this.data_ctx.stroke();

        this.data_ctx.closePath();
    }
}


function save_image(cvs, filename) {
    let a = document.createElement("a");
    a.setAttribute("download", filename);
    a.setAttribute("href", cvs.toDataURL("image/jpg"));
    a.click();
    a.remove();
}


let preview = document.getElementById("preview");
let mask_cvs = document.getElementById("mask");
let hints_cvs = document.getElementById("hints");
let illustration_cvs = document.getElementById("illustration");
let illustration_data_cvs = document.createElement("canvas");

mask_cvs.width = hints_cvs.width = illustration_cvs.width = illustration_data_cvs.width = IMG_W;
mask_cvs.height = hints_cvs.height = illustration_cvs.height = illustration_data_cvs.height = IMG_H;

let mask_ctx = mask_cvs.getContext("2d");
let hints_ctx = hints_cvs.getContext("2d");
let illustration_ctx = illustration_cvs.getContext("2d");
let illustration_data_ctx = illustration_data_cvs.getContext("2d");

let mask_draw_cvs = new DrawingCanvas(
    mask_cvs, mask_ctx, "#000000", "#ffffff", "#ffffff", 8, Tools.PEN
);
let hints_draw_cvs = new DrawingCanvas(
    hints_cvs, hints_ctx, "#ffffffff", "#000000", "#000000", 8, Tools.PEN
);


function set_size(size) { hints_draw_cvs.brush.size = mask_draw_cvs.brush.size = size; };
function set_tool(tool) { hints_draw_cvs.brush.tool = mask_draw_cvs.brush.tool = tool; };
function pick_color(element) {
    let input = document.createElement("input");
    input.addEventListener("change", event => {
        element.style.backgroundColor = event.target.value;
        hints_draw_cvs.brush.color = event.target.value;
    });
    input.setAttribute("type", "color");
    input.click();
}


let session = new onnx.InferenceSession({ backendHint: "webgl" });
session.loadModel("resources/paintstorch.onnx");

draw = () => {
    paintstorch(
        session,
        illustration_data_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
        mask_draw_cvs.data_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
        resize(hints_draw_cvs.data_cvs, IMG_W, IMG_H, IMG_W / 4, IMG_H / 4).data,
        illustration_ctx
    )
};

mask_draw_cvs.cvs.addEventListener("mouseup", event => draw());
hints_draw_cvs.cvs.addEventListener("mouseup", event => draw());


function upload(ctx, data_ctx) {
    let input = document.createElement("input");
    input.setAttribute("type", "file");
    input.addEventListener("change", () => {
        let file = input.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = event => {
                let img = new Image();
                img.onload = event => {
                    ctx.drawImage(img, 0, 0, IMG_W, IMG_H);
                    data_ctx.drawImage(img, 0, 0, IMG_W, IMG_H);
                    
                    preview.style.backgroundImage = "url(" + img.src + ")";
                    preview.style.backgroundSize = "100% 100%";
                    
                    draw();
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
    input.click();
}