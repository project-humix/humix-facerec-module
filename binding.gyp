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
      "libraries": [ "-Wl",
         "<!@(pkg-config opencv --libs)"
      ]
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
