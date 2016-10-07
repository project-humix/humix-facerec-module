var config = require('./lib/config');
var FaceRec = require('./lib/facerec').FaceRec;

var hfr = new FaceRec(config);

// constant
var threshold = 20;

var prevX;
var prevY;



setInterval(function() {

    var result = hfr.detect();
    console.log('result:' + JSON.stringify(result));

    if (result && result.pos_x && result.pos_y) {

        var newX = result.pos_x;
        var newY = result.pos_y;

        var deltaX = newX - prevX;
        var deltaY = newY - prevY;

        if (Math.abs(deltaX) > threshold) {

            var direction = deltaX > 0 ? "right" : "left";
            console.log("moving head to " + direction + " , distance" + Math.abs(deltaX));

        }


        if (Math.abs(deltaY) > threshold) {

            var direction = deltaY > 0 ? "down" : "up";
            console.log("moving head to " + direction + " , distance" + Math.abs(deltaY));

        }


        console.log('updating x and y');
        prevX = newX;
        prevY = newY;

    }

}, 5000);