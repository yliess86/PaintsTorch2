const IMG_W   = 512;
const IMG_H   = 512;
const SQR_IMG = IMG_W * IMG_H;


paintstorch = async function(session, input, mask, hints, ctx) {
    let input_tensor = new Tensor(new Float32Array(4 * IMG_H * IMG_W), "float32", [1, 4, IMG_H, IMG_W]);
    let hints_tensor = new Tensor(new Float32Array(4 * IMG_H / 4 * IMG_W / 4), "float32", [1, 4, IMG_H / 4, IMG_W / 4]);

    for(let y = 0; y < IMG_H; y++) for(let x = 0; x < IMG_W; x++) {
        let pos = (y * IMG_W + x) * 4;

        input_tensor[(y * IMG_W + x) + 0 * SQR_IMG] = input[pos + 0] / 255.0;
        input_tensor[(y * IMG_W + x) + 1 * SQR_IMG] = input[pos + 1] / 255.0;
        input_tensor[(y * IMG_W + x) + 2 * SQR_IMG] = input[pos + 2] / 255.0;
        input_tensor[(y * IMG_W + x) + 3 * SQR_IMG] = mask[pos + 0] / 255.0;
    }

    for(let y = 0; y < IMG_H / 4; y++) for(let x = 0; x < IMG_W / 4; x++) {
        let pos = (y * IMG_W / 4 + x) * 4;

        hints_tensor[(y * IMG_W/ 4 + x) + 0 * SQR_IMG] = hints[pos + 0] / 255.0;
        hints_tensor[(y * IMG_W/ 4 + x) + 1 * SQR_IMG] = hints[pos + 1] / 255.0;
        hints_tensor[(y * IMG_W/ 4 + x) + 2 * SQR_IMG] = hints[pos + 2] / 255.0;
        hints_tensor[(y * IMG_W/ 4 + x) + 3 * SQR_IMG] = hints[pos + 3] / 255.0;
    }


    session.run([input_tensor, hints_tensor]).then(output => {
        const tensor = output.values().next().value;
        const data = tensor.data;
        
        let buffer = new Uint8ClampedArray(IMG_W * IMG_H * 4);
        let idata = ctx.createImageData(IMG_H, IMG_W);
        
        for(let y = 0; y < IMG_H; y++) for(let x = 0; x < IMG_W; x++) {
            let pos = (y * IMG_W + x) * 4;

            let m = mask[pos + 0] / 255.0;
            buffer[pos + 0] = ((input[pos + 0] / 255.0) * (1 - m) + data[(y * IMG_W + x) + 0 * SQR_IMG] * m) * 255;
            buffer[pos + 1] = ((input[pos + 1] / 255.0) * (1 - m) + data[(y * IMG_W + x) + 1 * SQR_IMG] * m) * 255;
            buffer[pos + 2] = ((input[pos + 2] / 255.0) * (1 - m) + data[(y * IMG_W + x) + 2 * SQR_IMG] * m) * 255;
            buffer[pos + 3] = 255;
        }

        idata.data.set(buffer);
        ctx.putImageData(idata, 0, 0);
    });
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
        this.ctx.clearRect(0, 0, this.cvs.width, this.cvs.height);
        this.ctx.fillStyle = this.clear_color;
        this.ctx.fillRect(0, 0, this.cvs.width, this.cvs.height);
    }

    clear_data() {
        this.data_ctx.fillStyle = this.clear_color;
        this.data_ctx.fillRect(0, 0, this.data_cvs.width, this.data_cvs.height);
        this.clear();
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


save_image = function(cvs, filename) {
    let a = document.createElement("a");
    a.setAttribute("download", filename);
    a.setAttribute("href", cvs.toDataURL("image/jpg"));
    a.click();
    a.remove();
}


let mask_cvs = document.getElementById("mask");
let hints_cvs = document.getElementById("hints");
let illustration_cvs = document.getElementById("illustration");

mask_cvs.width = IMG_W;
mask_cvs.height = IMG_H;
hints_cvs.width = IMG_W;
hints_cvs.height = IMG_H;

let mask_ctx = mask_cvs.getContext("2d");
let hints_ctx = hints_cvs.getContext("2d");
let illustration_ctx = illustration_cvs.getContext("2d");

let mask_draw_cvs = new DrawingCanvas(
    mask_cvs, mask_ctx, "#000000", "#ffffff", "#ffffff", 5, Tools.PEN
);
let hints_draw_cvs = new DrawingCanvas(
    hints_cvs, hints_ctx, "#ffffffff", "#000000", "#000000", 5, Tools.PEN
);


set_size = function(size, draw_cvs) { draw_cvs.brush.size = size; };
set_tool = function(tool, draw_cvs) { draw_cvs.brush.tool = tool; };


let session = new onnx.InferenceSession({ backendHint: "webgl" });
let draw = () => paintstorch(
    session,
    illustration_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
    mask_draw_cvs.data_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
    hints_draw_cvs.data_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
    illustration_ctx
);

session.loadModel("resources/paintstorch.onnx");
mask_draw_cvs.cvs.addEventListener("mouseup", event => draw());
hints_draw_cvs.cvs.addEventListener("mouseup", event => draw());