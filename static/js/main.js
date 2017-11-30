drawing = false
function initialize() {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, 449, 449);
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, 449, 449);
    ctx.lineWidth = 0.05;
    for (var i = 0; i < 27; i++) {
        ctx.beginPath();
        ctx.moveTo((i + 1) * 16,   0);
        ctx.lineTo((i + 1) * 16, 449);
        ctx.closePath();
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(  0, (i + 1) * 16);
        ctx.lineTo(449, (i + 1) * 16);
        ctx.closePath();
        ctx.stroke();
    }
    drawInput();
    $('#output td').text('').removeClass('success');
}
function onMouseDown(e) {
    initialize()
    canvas.style.cursor = 'crosshair';
    drawing = true;
    prev = getPosition(e.clientX, e.clientY);
}
function onMouseUp() {
    drawing = false;
    canvas.style.cursor = 'default';
    drawInput();
}
function onMouseMove(e) {
    if (drawing) {
        var curr = getPosition(e.clientX, e.clientY);
        ctx.lineWidth = 36;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.stroke();
        ctx.closePath();
        prev = curr;
    }
}
function getPosition(clientX, clientY) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}
function drawInput() {
    var ctx = input.getContext('2d');
    var img = new Image();
    img.onload = () => {
        var inputs = [];
        var small = document.createElement('canvas').getContext('2d');
        small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
        var data = small.getImageData(0, 0, 28, 28).data;
        for (var i = 0; i < 28; i++) {
            for (var j = 0; j < 28; j++) {
                var n = 4 * (i * 28 + j);
                inputs[i * 28 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                ctx.fillRect(j * 5, i * 5, 5, 5);
            }
        }
        if (Math.min(...inputs) === 255) {
            return;
        }
        $.ajax({
            url: '/api/mnist',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: (data) => {
                for (let i = 0; i < 4; i++) {
                    var max = 0;
                    var max_index = 0;
                    for (let j = 0; j < 10; j++) {
                        var value = Math.round(data.results[i][j] * 1000);
                        if (value > max) {
                            max = value;
                            max_index = j;
                        }
                        var digits = String(value).length;
                        for (var k = 0; k < 3 - digits; k++) {
                            value = '0' + value;
                        }
                        var text = '0.' + value;
                        if (value > 999) {
                            text = '1.000';
                        }
                        $('#output tr').eq(j + 1).find('td').eq(i).text(text);
                    }
                    for (let j = 0; j < 10; j++) {
                        if (j === max_index) {
                            $('#output tr').eq(j + 1).find('td').eq(i).addClass('success');
                        } else {
                            $('#output tr').eq(j + 1).find('td').eq(i).removeClass('success');
                        }
                    }
                }
            }
        });
    };
    img.src = canvas.toDataURL();
}
function get_stats(){
    //to reduce bandwidth usage under slow connection speeds
    if (is_polling)
        return
    is_polling = true

    $.get("/visit_stats",{},function(result){
        result = result.results;
        $("#visits").html(result.visits);
        $("#uniq_visits").html(result.uniq_visits);
        $("#prediction_reqs").html(result.prediction_reqs);
        is_polling = false
    });
}
is_polling = false
canvas = document.getElementById('main');
input = document.getElementById('input');
canvas.width  = 449; // 16 * 28 + 1
canvas.height = 449; // 16 * 28 + 1
ctx = canvas.getContext('2d');
canvas.addEventListener('mousedown', onMouseDown.bind(this));
canvas.addEventListener('mouseup',   onMouseUp.bind(this));
canvas.addEventListener('mousemove', onMouseMove.bind(this));
initialize();
setInterval(get_stats, 5000);
get_stats()
