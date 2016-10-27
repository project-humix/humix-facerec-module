/*******************************************************************************
 * Copyright (c) 2015 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
'use strict';

var fs = require('fs');
var HumixSense = require('humix-sense');
var path = require('path');
var _ = require('lodash');

var config = require('./lib/config');
var FaceRec = require('./lib/facerec').FaceRec;

var moduleConfig = {
    moduleName: 'facerec',
    commands: [
        'train',
        'detect-face',
        'no-detect',
        'detect-motion',
        'track-face',
        'no-track'
    ],
    events: [
        'detectComplete',
        'faceDetected',
        'trainComplete',
        'facePosition',
        'motion-detected',
        'face-right',
        'face-left',
        'face-up',
        'face-down'
    ],
    debug: true
}






var humix = new HumixSense(moduleConfig);
var hsm;
var hfr = new FaceRec(config);
var trainedImg = path.resolve(__dirname, 'images', 'orig', 'trained.jpg');
var origImgRoot = path.resolve(__dirname, 'images', 'orig');

// long running events
var detecting;
var detectingMotion;
var faceTrackingTask;

var detectedHistory = {};

humix.on('connection', function(humixSensorModule) {

    hsm = humixSensorModule;
    console.log('Communication with humix-sense is now ready.');

    hsm.on('train', function(data) {
        data = JSON.parse(data);
        if (hfr.captureAndTrain(data.name, 20)) {

            var image = fs.readFileSync(trainedImg).toString('base64');
            hsm.event('trainComplete',
                JSON.stringify({
                    id: data.id,
                    image: image
                }));

        } else {
            //failed to perform the training
            hsm.event('trainComplete',
                JSON.stringify({
                    id: data.id,
                    msg: 'failed'
                }));
        }
    });

    hsm.on('detect-face', function(data) {
        data = JSON.stringify(data);
        console.log('start detecting face');
        detecting = setInterval(doDetect, 1000);
    });


    hsm.on('no-detect', function(data) {

        console.log('disable face detection');
        if (detecting) {

            clearInterval(detecting);
            detecting = undefined;
        }

    });

    hsm.on('detect-motion', function(data) {

        console.log('start detecting motion');

        if (detectingMotion) {

            clearInterval(detectingMotion);
            detectingMotion = undefined;
        }

        //detectingMotion = setInterval

    });

    hsm.on('track-face', function(data) {

        console.log('start tracking face');

        doFaceTracking();


    });

    hsm.on('no-track', function(data) {

        console.log('disable face tracking');
        if (faceTrackingTask) {

            clearInterval(faceTrackingTask);
            faceTrackingTask = undefined;
        }

    });


});


var facePrevX = -1;
var facePrevY = -1;
var threshold = 50;

var baseX = 300;
var baseY = 260;

function doFaceTracking() {

    if (faceTrackingTask) {

        console.log('disable previous face tracking task');
        clearInterval(detectingMotion);
        faceTrackingTask = null;
    }

    faceTrackingTask = setInterval(function() {

        var result = hfr.track();
        //console.log('result -- ' + JSON.stringify(result));

        if (result && result.pos_x && result.pos_y) {



            var newX = result.pos_x;
            var newY = result.pos_y;

            console.log('old x:' + facePrevX + ", new x:" + newX);
            console.log('old y:' + facePrevY + ", new x:" + newY);
            var deltaX = newX - baseX;
            var deltaY = newY - baseY;

            if (Math.abs(deltaX) > threshold) {

                var direction = deltaX > 0 ? "right" : "left";
                console.log("moving head to " + direction + " , distance" + Math.abs(deltaX));


                hsm.event(deltaX > 0 ? 'face-right' : 'face-left',
                    JSON.stringify({
                        posX: newX,
                        deltaX: deltaX
                    }));

                console.log('updating x ');
                facePrevX = newX;

            }


            if (Math.abs(deltaY) > threshold) {

                var direction = deltaY > 0 ? "down" : "up";
                console.log("moving head to " + direction + " , distance" + Math.abs(deltaY));

                hsm.event(deltaY > 0 ? 'face-down' : 'face-up',
                    JSON.stringify({
                        posY: newY,
                        deltaY: deltaY
                    }));

                facePrevY = newY;
            }

        }

    }, 2000);
}

function doDetect() {
    var result = hfr.detect();
    if (!_.isUndefined(result)) {
        //see if we detect this person within 30 seconds
        try {
            var historyEntry = detectedHistory[result.name];
            //if (_.isUndefined(historyEntry) || (Date.now() - historyEntry) > 30000) {
            detectedHistory[result.name] = Date.now();
            //var img = fs.readFileSync(path.resolve(origImgRoot, result.name + '.jpg')).toString('base64');

            console.log('face detected. Pos:' + result.pos_x + ',' + result.pos_y);
            hsm.event('faceDetected',
                JSON.stringify({
                    name: result.name,
                    conf: result.conf,
                    pos_x: result.pos_x,
                    pos_y: result.pos_y
                        //image: img
                }));
            //}
        } catch (e) {
            console.error(e);
        }
    }
}