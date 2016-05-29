{
  "targets": [
    {
      "target_name": "HumixFaceRec",
      "sources": [
        "./src/HumixFaceRec.cpp"
      ],
      "include_dirs": [ "<!(node -e \"require('nan')\")",
        "/usr/local/include/opencv2/",
        "<!@(pkg-config opencv --cflags-only-I | sed s/-I//g)"
      ],
      "libraries": [
         "<!@(pkg-config opencv --libs)"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ]
    },
    {
      "target_name": "action_after_build",
      "type": "none",
      "dependencies": [ "HumixFaceRec" ],
      "copies": [{
        "destination": "./lib/",
        "files": [
          "<(PRODUCT_DIR)/HumixFaceRec.node"
        ]},
      ]
    }

  ]
}
