const ops = ndarray.ops;


class PaintsTorch2 {
    constructor() {
        this.model = undefined;

        this.WIDTH  = 512;
        this.HEIGHT = 512;
        this.PIXELS = this.WIDTH * this.HEIGHT;
        
        this.H_WIDTH  = 128;
        this.H_HEIGHT = 128;
        this.H_PIXELS = this.H_WIDTH * this.H_HEIGHT;
        
        this.load();
    }

    load = async () => {
        this.model = await tf.loadGraphModel("resources/paintstorch2/model.json");
        const x = tf.zeros([1, 4, this.HEIGHT, this.WIDTH]);
        const h = tf.zeros([1, 4, this.H_HEIGHT, this.H_WIDTH]);
        await this.model.executeAsync({ "input": x, "hints": h }, "Identity");
    }

    draw = (x_ctx, m_ctx, h_ctx, callback) => this.forward(
        x_ctx.getImageData(0, 0, this.WIDTH, this.HEIGHT).data,
        m_ctx.getImageData(0, 0, this.WIDTH, this.HEIGHT).data,
        h_ctx.getImageData(0, 0, this.WIDTH, this.HEIGHT).data,
    ).then(data => callback(data));

    to_tensor = data => tf.tensor3d(new Float32Array(data, [this.HEIGHT, this.WIDTH, 4]), [this.HEIGHT, this.WIDTH, 4], "float32");
    to_data = tensor => new ImageData(Uint8ClampedArray.from(tensor), this.WIDTH, this.HEIGHT);

    forward = async (x_data, m_data, h_data) => {
        if (this.model == undefined) return new ImageData(x, this.WIDTH, this.HEIGHT);

        let x = this.to_tensor(x_data);
        let m = this.to_tensor(m_data);
        let h = this.to_tensor(h_data).resizeBilinear([128, 128]);

        x = tf.stack([
            tf.gather(x, 0, 2).div(255.0).sub(0.5).div(0.5),
            tf.gather(x, 1, 2).div(255.0).sub(0.5).div(0.5),
            tf.gather(x, 2, 2).div(255.0).sub(0.5).div(0.5),
            tf.gather(m, 0, 2).div(255.0),
        ], 2).transpose([2, 0, 1]).expandDims(0);

        h = tf.stack([
            tf.gather(h, 0, 2).div(255.0).sub(0.5).div(0.5),
            tf.gather(h, 1, 2).div(255.0).sub(0.5).div(0.5),
            tf.gather(h, 2, 2).div(255.0).sub(0.5).div(0.5),
            tf.gather(h, 3, 2).div(255.0),
        ], 2).transpose([2, 0, 1]).expandDims(0);

        let y = await this.model.executeAsync({ "input": x, "hints": h }, "Identity");
        y = y.squeeze(0).transpose([1, 2, 0])
        y = y.mul(0.5).add(0.5);
        y = tf.concat([y, tf.fill([512, 512, 1], 1.0)], 2).mul(255.0);
        y = await y.dataSync();

        return this.to_data(y);
    };
}


const Tools = { PEN: "pen", ERASER: "eraser", COLOR_PICKER: "color-picker" };
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

        if (this.tool == Tools.COLOR_PICKER) {
            ctx.beginPath();

            ctx.fillStyle = "rgb(" + value + "," + value + "," + value + ")";
            ctx.fillRect(position.x + 1, position.y + 1, -27, -27);

            ctx.fillStyle = "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
            ctx.fillRect(position.x, position.y, -25, -25);

            ctx.closePath();
        }
    };

    get_position = (cvs, event) => {
        let rect = cvs.getBoundingClientRect();
        return new Position(event.clientX - rect.left, event.clientY - rect.top);
    };
}


class DrawingCVS {
    constructor(id, clear_color, data_clear_color, brush_color, bckg_cvs, color_viewer) {
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

        this.color_viewer = color_viewer;
        
        this.brush = new Brush(this.BRUSH_COLOR);
        this.in = false;

        this.display_cvs.addEventListener("mousemove", event => this.update(event));

        this.display_cvs.addEventListener("mousedown", () => {
            if(this.brush.tool != Tools.COLOR_PICKER) this.brush.enable();
        });

        this.display_cvs.addEventListener("mouseup", () => {
            if(this.brush.tool == Tools.COLOR_PICKER) {
                const position = this.brush.position.current;
                const color = this.data_ctx.getImageData(position.x, position.y, 1, 1).data;

                this.brush.color = "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
                this.color_viewer.style.backgroundColor = "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
            }
            this.brush.disable();
        });
        
        this.display_cvs.addEventListener("mouseenter", () => {
            this.in = true;
            this.update();
        });

        this.display_cvs.addEventListener("mouseout", () => {
            this.in = false;
            this.brush.disable();
            this.update();
        });
        
        this.clear();
        this.update();
    }

    draw = () => {
        if (!this.brush.active) return;

        this.data_ctx.beginPath();
        
        const is_pen = this.brush.tool == Tools.PEN;
        const is_plain = this.DATA_CLEAR_COLOR[8] == "f";
        
        this.data_ctx.globalCompositeOperation = is_pen? "source-over": is_plain? "source-over": "destination-out";
        this.data_ctx.strokeStyle = is_pen? this.brush.color: is_plain? this.DATA_CLEAR_COLOR: null;
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

        this.bckg_cvs = document.createElement("canvas");
        this.bckg_ctx = this.bckg_cvs.getContext("2d");
        this.bckg_cvs.width = this.bckg_cvs.height = this.SIZE;

        this.in = false;

        this.display_cvs.addEventListener("mouseenter", () => {
            this.in = true;
            this.update();
        });
        this.display_cvs.addEventListener("mouseout", () => {
            this.in = false;
            this.update();
        });

        this.clear();
        this.update();
    }

    update = () => {
        this.display_ctx.fillStyle = this.CLEAR_COLOR;
        this.display_ctx.fillRect(0, 0, this.SIZE, this.SIZE);
        this.display_ctx.drawImage(this.in? this.bckg_cvs: this.data_cvs, 0, 0);
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
        let promise = new Promise((resolve, reject) => {
            let file = input.files[0];
            if (file) {
                let reader = new FileReader();
                reader.onload = event => {
                    let img = new Image();
                    img.onload = () => {
                        cvs.clear();
                        ctx.drawImage(img, 0, 0, 512, 512);
                        cvs.update();
                        resolve();
                    };
                    img.src = event.target.result;
                };
                reader.readAsDataURL(file);
            }
        });
        promise.then(callback);
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
let color_picker_btn = document.getElementById("color-picker-btn");
let size_btns = {
    TINY  : document.getElementById("size-tiny-btn"),
    SMALL : document.getElementById("size-small-btn"),
    MEDIUM: document.getElementById("size-medium-btn"),
    BIG   : document.getElementById("size-big-btn"),
    HUGE  : document.getElementById("size-huge-btn"),
};

let illustration_dcvs = new DisplayCVS("illustration");
let bckg_cvs = illustration_dcvs.bckg_cvs;

let mask_dcvs = new DrawingCVS("mask", "#000000ff", "#000000ff", "#ffffffff", bckg_cvs, color_btn);
let hints_dcvs = new DrawingCVS("hints", "#000000ff", "#00000000", "#ffffffff", bckg_cvs, color_btn);

let paintstorch = new PaintsTorch2();
let paint = () => paintstorch.draw(illustration_dcvs.bckg_ctx, mask_dcvs.data_ctx, hints_dcvs.data_ctx, data => {
    illustration_dcvs.data_ctx.putImageData(data, 0, 0);
    
    illustration_dcvs.update();
    mask_dcvs.update();
    hints_dcvs.update();
});

mask_dcvs.display_cvs.addEventListener("mouseup", () => { if(mask_dcvs.brush.tool != Tools.COLOR_PICKER) paint(); });
hints_dcvs.display_cvs.addEventListener("mouseup", () => { if(mask_dcvs.brush.tool != Tools.COLOR_PICKER) paint(); });

mask_upload_btn.addEventListener("click", () => upload(mask_dcvs.data_ctx, mask_dcvs, paint));
hints_upload_btn.addEventListener("click", () => upload(hints_dcvs.data_ctx, hints_dcvs, paint));
illustration_upload_btn.addEventListener("click", () => {
    upload(illustration_dcvs.data_ctx, illustration_dcvs, () => {
        mask_dcvs.bckg_ctx.drawImage(illustration_dcvs.data_cvs, 0, 0, 512, 512);
        hints_dcvs.bckg_ctx.drawImage(illustration_dcvs.data_cvs, 0, 0, 512, 512);
        illustration_dcvs.bckg_ctx.drawImage(illustration_dcvs.data_cvs, 0, 0, 512, 512);
        paint();
    });
});

mask_save_btn.addEventListener("click", () => save(mask_dcvs.data_cvs, "mask.png"));
hints_save_btn.addEventListener("click", () => save(hints_dcvs.data_cvs, "hints.png"));
illustration_save_btn.addEventListener("click", () => save(illustration_dcvs.data_cvs, "illustration.png"));

mask_clean_btn.addEventListener("click", () => {
    mask_dcvs.clear();
    paint();
});
hints_clean_btn.addEventListener("click", () => {
    hints_dcvs.clear();
    paint();
});

mask_fill_btn.addEventListener("click", () => {
    mask_dcvs.fill();
    paint();
});

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
    eraser_btn.classList.remove("active");
    color_picker_btn.classList.remove("active");
    mask_dcvs.brush.tool = hints_dcvs.brush.tool = Tools.PEN;
});
eraser_btn.addEventListener("click", () => {
    eraser_btn.classList.add("active");
    pen_btn.classList.remove("active");
    color_picker_btn.classList.remove("active");
    mask_dcvs.brush.tool = hints_dcvs.brush.tool = Tools.ERASER;
});
color_picker_btn.addEventListener("click", () => {
    color_picker_btn.classList.add("active");
    pen_btn.classList.remove("active");
    eraser_btn.classList.remove("active");
    mask_dcvs.brush.tool = hints_dcvs.brush.tool = Tools.COLOR_PICKER;
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