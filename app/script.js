class PaintsTorch2 {
    constructor() {
        this.session = new onnx.InferenceSession({ backendHint: "webgl" });
        this.session.loadModel("resources/paintstorch.onnx");

        this.WIDTH  = 512;
        this.HEIGHT = 512;
        this.PIXELS = this.WIDTH * this.HEIGHT;

        this.H_WIDTH  = 128;
        this.H_HEIGHT = 128;
        this.H_PIXELS = this.H_WIDTH * this.H_HEIGHT;
    }

    forward = async (x, m, h) => {
        let x_from = ndarray(new Float32Array(input), [this.HEIGHT,   this.WIDTH,   4]);
        let m_from = ndarray(new Float32Array(input), [this.HEIGHT,   this.WIDTH,   4]);
        let h_from = ndarray(new Float32Array(input), [this.H_HEIGHT, this.H_WIDTH, 4]);

        let x_to = ndarray(new Float32Array(this.PIXELS   * 4), [1, 4, this.HEIGHT,   this.WIDTH  ]);
        let m_to = ndarray(new Float32Array(this.PIXELS   * 1), [1,    this.HEIGHT,   this.WIDTH  ]);
        let h_to = ndarray(new Float32Array(this.H_PIXELS * 4), [1, 4, this.H_HEIGHT, this.H_WIDTH]);

        ndarray.ops.assign(x_from.pick(null, null, 3), m_from.pick(null, null, 0));
        ndarray.ops.assign(m_to.pick(0, null, null), m_from.pick(null, null, 0));
        ndarray.ops.divseq(m_to.pick(0, null, null), 255);

        for(let i = 0; i < 4; i++) {
            ndarray.ops.assign(x_to.pick(0, i, null, null), x_from.pick(null, null, i));
            ndarray.ops.assign(h_to.pick(0, i, null, null), h_from.pick(null, null, i));
            ndarray.ops.divseq(x_to.pick(0, i, null, null), 255);
            ndarray.ops.divseq(h_to.pick(0, i, null, null), 255);
            
            if(i < 3) {
                ndarray.ops.subseq(x_to.pick(0, i, null, null), 0.5);
                ndarray.ops.subseq(h_to.pick(0, i, null, null), 0.5);
                ndarray.ops.divseq(x_to.pick(0, i, null, null), 0.5);
                ndarray.ops.divseq(h_to.pick(0, i, null, null), 0.5);
            }
        }

        let x_tensor = new Tensor(x_to.data, "float32", [1, 4, this.HEIGHT,   this.WIDTH  ]);
        let m_tensor = new Tensor(m_to.data, "float32", [1,    this.HEIGHT,   this.WIDTH  ]);
        let h_tensor = new Tensor(h_to.data, "float32", [1, 4, this.H_HEIGHT, this.H_WIDTH]);
        
        const y_map  = await this.session.run([x_tensor, h_tensor, m_tensor]);
        const y_data = y_map.values().next().value.data;

        let y_from = ndarray(new Float32Array(y_data), [1, 3, this.HEIGHT, this.WIDTH]);
        let y_to   = ndarray(new Float32Array(this.PIXELS * 4), [this.HEIGHT, this.WIDTH, 4]);
        
        for(let i = 0; i < 3; i++) {
            ndarray.ops.mulseq(y_from.pick(0, i, null, null), 0.5);
            ndarray.ops.addseq(y_from.pick(0, i, null, null), 0.5);
            ndarray.ops.mulseq(y_from.pick(0, i, null, null), 255);
            ndarray.ops.assign(y_to.pick(null, null, i), y_from.pick(0, i, null, null));
        }

        ndarray.ops.assigns(y_to.pick(null, null, 3), 255);

        return new ImageData(Uint8ClampedArray.from(y_to.data), this.WIDTH, this.HEIGHT);
    };
}


const Tools = { PEN: "pen", ERASER: "eraser" };
const Sizes = { TINY: 2, SMALL: 8, MEDIUM: 16, BIG: 32, HUGE: 64 };


class Position {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}


class Brush {
    constructor(color) {
        this.position = { current: new Position(0, 0), last: new Position(0, 0)};
        this.size     = Sizes.SMALL;
        this.tool     = Tools.PEN;
        this.color    = color;
        this.active   = false;
    }

    enable  = () => this.active = true;
    disable = () => this.active = false;
    
    draw = (ctx) => {
        ctx.beginPath();
        
        const position = this.position.current;
        const color = ctx.getImageData(position.x, position.y, 1, 1).data;
        const value = ((color[0] + color[1] + color[2]) / 3) > 128 ? 0: 255;
        
        ctx.fillStyle = null;
        ctx.strokeStyle = "rgb(" + value + "," + value + "," + value + ")";
        ctx.arc(position.x, position.y, this.size, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.closePath();
    };

    get_position = (cvs, event) => {
        let rect = cvs.getBoundingClientRect();
        return new Position(event.clientX - rect.left, event.clientY - rect.top);
    };
}


class DrawingCVS {
    constructor(id, clear_color, data_clear_color, brush_color, bckg_cvs) {
        this.CLEAR_COLOR = clear_color;
        this.DATA_CLEAR_COLOR = data_clear_color;
        this.BRUSH_COLOR = brush_color;
        this.SIZE = 512;

        this.display_cvs = document.getElementById(id);
        this.display_ctx = this.display_cvs.getContext("2d");
        this.display_cvs.width = this.display_cvs.height = this.SIZE;
        
        this.data_cvs = document.createElement("canvas");
        this.data_ctx = this.data_cvs.getContext("2d");
        this.data_cvs.width = this.data_cvs.height = this.SIZE;

        this.bckg_cvs = bckg_cvs;
        this.bckg_ctx = this.bckg_cvs.getContext("2d");
        
        this.brush = new Brush(this.BRUSH_COLOR);
        this.in = false;

        this.display_cvs.addEventListener("mousemove", event => this.update(event));
        this.display_cvs.addEventListener("mousedown",    () => this.brush.enable());
        this.display_cvs.addEventListener("mouseup",      () => this.brush.disable());
        
        this.display_cvs.addEventListener("mouseenter", () => {
            this.in = true;
            this.update();
        });

        this.display_cvs.addEventListener("mouseout", () => {
            this.in = false;
            this.brush.disable();
            this.update();
        });
        
        this.update();
    }

    draw = () => {
        if (!this.brush.active) return;

        this.data_ctx.beginPath();
        
        const is_pen = this.brush.tool == Tools.PEN;
        
        this.data_ctx.strokeStyle = is_pen? this.brush.color: this.CLEAR_COLOR;
        this.data_ctx.lineWidth = this.brush.size;
        
        const current = this.brush.position.current;
        const last = this.brush.position.last;
        
        this.data_ctx.moveTo(current.x, current.y);
        this.data_ctx.lineTo(last.x, last.y);
        this.data_ctx.lineJoin = this.data_ctx.lineCap = "round";
        this.data_ctx.stroke();

        this.data_ctx.closePath();
    };

    update = (event) => {
        if (event != undefined)
            this.brush.position.current = this.brush.get_position(this.display_cvs, event);
        
        this.display_ctx.fillStyle = this.CLEAR_COLOR;
        this.display_ctx.fillRect(0, 0, this.SIZE, this.SIZE);

        this.draw();

        this.display_ctx.globalAlpha = this.in ? 0.5: 0.0;
        this.display_ctx.drawImage(this.bckg_cvs, 0, 0);
        
        this.display_ctx.globalAlpha = this.in ? 0.5: 1.0;
        this.display_ctx.drawImage(this.data_cvs, 0, 0);
        
        this.display_ctx.globalAlpha = this.in ? 1.0: 0.0;
        this.brush.draw(this.display_ctx);
        
        if (event != undefined)
            this.brush.position.last = this.brush.get_position(this.display_cvs, event);
    };
    
    clear = () => {
        this.data_ctx.clearRect(0, 0, this.SIZE, this.SIZE);
        this.data_ctx.fillStyle = this.DATA_CLEAR_COLOR;
        this.data_ctx.fillRect(0, 0, this.SIZE, this.SIZE);
        this.update();
    };

    fill = () => {
        this.data_ctx.fillStyle = this.BRUSH_COLOR;
        this.data_ctx.fillRect(0, 0, this.SIZE, this.SIZE);
        this.update();
    };
}


class DisplayCVS {
    constructor(id) {
        this.CLEAR_COLOR = "#ffffff";
        this.SIZE = 512;

        this.display_cvs = document.getElementById(id);
        this.display_ctx = this.display_cvs.getContext("2d");
        this.display_cvs.width = this.display_cvs.height = this.SIZE;
        
        this.data_cvs = document.createElement("canvas");
        this.data_ctx = this.data_cvs.getContext("2d");
        this.data_cvs.width = this.data_cvs.height = this.SIZE;
    }

    update = () => {
        this.display_ctx.fillStyle = this.CLEAR_COLOR;
        this.display_ctx.fillRect(0, 0, this.SIZE, this.SIZE);
        this.display_ctx.drawImage(this.data_cvs, 0, 0);
    };

    clear = () => {
        this.data_ctx.fillStyle = this.CLEAR_COLOR;
        this.data_ctx.fillRect(0, 0, this.SIZE, this.SIZE);
        this.update();
    };
}


let upload = (ctx, cvs, callback) => {
    let input = document.createElement("input");
    input.setAttribute("type", "file");
    input.addEventListener("change", () => {
        let file = input.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = event => {
                let img = new Image();
                img.onload = () => {
                    cvs.clear();
                    ctx.drawImage(img, 0, 0, 512, 512);
                    cvs.update();
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
    input.click();
};


let save = (cvs, filename) => {
    let a = document.createElement("a");
    a.setAttribute("download", filename);
    a.setAttribute("href", cvs.toDataURL("image/jpg"));
    a.click();
    a.remove();
};


let paintstorch = new PaintsTorch2();

let illustration_dcvs = new DisplayCVS("illustration");
let bckg_cvs = illustration_dcvs.data_cvs;

let mask_dcvs = new DrawingCVS("mask", "#000000ff", "#000000ff", "#ffffffff", bckg_cvs);
let hints_dcvs = new DrawingCVS("hints", "#000000ff", "#00000000", "#ffffffff", bckg_cvs);

let mask_upload_btn = document.getElementById("mask-upload-btn");
let hints_upload_btn = document.getElementById("hints-upload-btn");
let illustration_upload_btn = document.getElementById("illustration-upload-btn");

let mask_save_btn = document.getElementById("mask-save-btn");
let hints_save_btn = document.getElementById("hints-save-btn");
let illustration_save_btn = document.getElementById("illustration-save-btn");

let mask_clean_btn = document.getElementById("mask-clean-btn");
let hints_clean_btn = document.getElementById("hints-clean-btn");

let mask_fill_btn = document.getElementById("mask-fill-btn");

let color_btn = document.getElementById("color-btn");
let pen_btn = document.getElementById("pen-btn");
let eraser_btn = document.getElementById("eraser-btn");
let size_btns = {
    TINY  : document.getElementById("size-tiny-btn"),
    SMALL : document.getElementById("size-small-btn"),
    MEDIUM: document.getElementById("size-medium-btn"),
    BIG   : document.getElementById("size-big-btn"),
    HUGE  : document.getElementById("size-huge-btn"),
};

mask_upload_btn.addEventListener("click", () => upload(mask_dcvs.data_ctx, mask_dcvs));
hints_upload_btn.addEventListener("click", () => upload(hints_dcvs.data_ctx, hints_dcvs));
illustration_upload_btn.addEventListener("click", () => {
    upload(illustration_dcvs.data_ctx, illustration_dcvs, () => {
        mask_dcvs.bckg_ctx.drawImage(illustration_dcvs.data_cvs, 0, 0, 512, 512);
        hints_dcvs.bckg_ctx.drawImage(illustration_dcvs.data_cvs, 0, 0, 512, 512);
    });
});

mask_save_btn.addEventListener("click", () => save(mask_dcvs.data_cvs, "mask.png"));
hints_save_btn.addEventListener("click", () => save(hints_dcvs.data_cvs, "hints.png"));
illustration_save_btn.addEventListener("click", () => save(illustration_dcvs.data_cvs, "illustration.png"));

mask_clean_btn.addEventListener("click", () => mask_dcvs.clear());
hints_clean_btn.addEventListener("click", () => hints_dcvs.clear());

mask_fill_btn.addEventListener("click", () => mask_dcvs.fill());

color_btn.addEventListener("click", () => {
    let input = document.createElement("input");
    input.addEventListener("change", event => {
        hints_dcvs.brush.color = color_btn.style.backgroundColor = event.target.value;
    });
    input.setAttribute("type", "color");
    input.click();
});
pen_btn.addEventListener("click", () => {
    pen_btn.classList.add("active");
    eraser_btn.classList.add("active");
    mask_dcvs.brush.tool = hints_dcvs.brush.tool = Tools.PEN;
});
eraser_btn.addEventListener("click", () => {
    eraser_btn.classList.add("active");
    pen_btn.classList.add("active");
    mask_dcvs.brush.tool = hints_dcvs.brush.tool = Tools.ERASER;
});
Object.entries(size_btns).forEach(([key, btn]) => {
    btn.addEventListener("click", () => {
        mask_dcvs.brush.size = hints_dcvs.brush.size = Sizes[key];
        Object.entries(size_btns).forEach(([k, b]) => {
            if (k == key) b.classList.add("active");
            else b.classList.remove("active");
        });
    });
});


// function resize(scvs, sw, sh, nw, nh) {
//     let cvs = document.createElement("canvas");
//     let ctx = cvs.getContext("2d");
//     cvs.height = nh;
//     cvs.width = nw;

//     ctx.drawImage(scvs, 0, 0, sw, sh, 0, 0, nw, nh);
//     return ctx.getImageData(0, 0, nw, nh);
// }

// draw = () => {
//     paintstorch(
//         session,
//         illustration_data_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
//         mask_draw_cvs.data_ctx.getImageData(0, 0, IMG_W, IMG_H).data,
//         resize(hints_draw_cvs.data_cvs, IMG_W, IMG_H, IMG_W / 4, IMG_H / 4).data,
//         illustration_ctx
//     )
// };