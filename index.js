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
var console = require('console');
var config  = require('./lib/config');
var nats    = require('nats').connect();

var HumixSense = require('node-humix-sense');
var HumixFaceRec = require('./lib/HumixFaceRec').HumixFaceRec;

var moduleConfig = {
    "moduleName":"humix-facerec",
    "commands" : ["start","stop","train"],
    "events" : ["detect"],
    "debug": true
}

var humix = new HumixSense(moduleConfig);
var hsm;

humix.on('connection', function(humixSensorModule){

    hsm = humixSensorModule;

    console.log('Communication with humix-sense is now ready.');

    hsm.on('train', function(data){
        console.log('data:'+data);
        startTraining(data);
    });  // end of say command
});

function startTraining(){

    console.log('start trainning');
}


var hs;

/**
 * callback function that is called when
 * HumixSpeech detect a valid command/sentence
 * @param cmdstr a command/sentence in this format:
 *         '----="command string"=---'
 */
function detectFace() {


    console.log('face detected');
    /*
    cmdstr = cmdstr.trim();
    if ( config.engine ) {
        console.error('command found:', cmdstr);
        
        if(hsm)
            hsm.event("detect", cmdstr);

    } else {
        var match = commandRE.exec(cmdstr);
        if ( match && match.length == 2 ) {
            var cmd = match[1];
            console.error('command found:', cmd);
            try {
                nats.publish('humix.sense.speech.event', cmd);
            } catch ( e ) {
                console.error('can not publish to nats:', e);
            }
            //echo mode
            //text2Speech( '{ "text" : "' + cmd + '" }' );
            if ( hs && cmd.indexOf('聖誕') != -1 && cmd.indexOf('快樂') != -1 ) {
                hs.play('./voice/music/jingle_bells.wav');
            }
        }
    }

    */
}

try {

    hs = new HumixFaceRec(config.options);
    hs.start(detectFace);

} catch ( error ) {
    console.error(error);
}

function cleanup() {
    if (hs) {
        hs.stop();
    }
}

process.on('SIGINT', function() {
    cleanup();
    process.exit(0);
});
process.on('SIGHUP', function() {
    cleanup();
    process.exit(0);
});
process.on('SIGTERM', function() {
    cleanup();
    process.exit(0);
});
process.on('exit', function() {
    cleanup();
});
process.on('error', function() {
    cleanup();
});

process.on('uncaughtException', function(err) {
    if ( err.toString().indexOf('connect ECONNREFUSED') ) {
        console.error('exception,', JSON.stringify(err));
    }
});

