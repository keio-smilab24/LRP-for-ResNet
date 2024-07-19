# ## Main Experiment
# ### skip-connection-prop-type = "flows_skip" (proposed)
# poetry run python visualize.py -c configs/ImageNet_resnet50.json --method "lrp" --heat-quantization --skip-connection-prop-type "flows_skip" --notes "imagenet--type:flows_skip--viz:norm+positive" --all_class --seed 42 --normalize --sign "positive"


# ## Visualize Conservation
# ### see & modify also: src/lrp.py:L146-L159
# ### skip-connection-prop-type = "flows_skip" (proposed)
# poetry run python visualize.py -c configs/ImageNet_resnet50.json --method "lrp" --heat-quantization --skip-connection-prop-type "flows_skip" --notes "visualize-conservation@input" --all_class --seed 42 --normalize --sign "positive" --data_limit 100


# ## Visualize attribution maps for specific images
# ### CUB
# ### Savannah_Sparrow
# ### lrp
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumlrp" \
#     --image-path ./qual/original/Savannah_Sparrow.png --label 126 --save-path ./qual/lrp/Savannah_Sparrow.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumig" \
#     --image-path ./qual/original/Savannah_Sparrow.png --label 126 --save-path ./qual/ig/Savannah_Sparrow.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/Savannah_Sparrow.png --label 126 --save-path ./qual/guidedpb/Savannah_Sparrow.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "gradcam" \
#     --image-path ./qual/original/Savannah_Sparrow.png --label 126 --save-path ./qual/gradcam/Savannah_Sparrow.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "scorecam" \
#     --image-path ./qual/original/Savannah_Sparrow.png --label 126 --save-path ./qual/scorecam/Savannah_Sparrow.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/Savannah_Sparrow.png --label 126 --save-path ./qual/ours/Savannah_Sparrow.png --normalize --sign "positive"

# ### Brandt_Cormorant
# ### lrp
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumlrp" \
#     --image-path ./qual/original/Brandt_Cormorant.png --label 22 --save-path ./qual/lrp/Brandt_Cormorant.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumig" \
#     --image-path ./qual/original/Brandt_Cormorant.png --label 22 --save-path ./qual/ig/Brandt_Cormorant.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/Brandt_Cormorant.png --label 22 --save-path ./qual/guidedpb/Brandt_Cormorant.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "gradcam" \
#     --image-path ./qual/original/Brandt_Cormorant.png --label 22 --save-path ./qual/gradcam/Brandt_Cormorant.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "scorecam" \
#     --image-path ./qual/original/Brandt_Cormorant.png --label 22 --save-path ./qual/scorecam/Brandt_Cormorant.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/Brandt_Cormorant.png --label 22 --save-path ./qual/ours/Brandt_Cormorant.png --normalize --sign "positive"

# ### Rock_Wren
# ### lrp
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumlrp" \
#     --image-path ./qual/original/Rock_Wren.png --label 197 --save-path ./qual/lrp/Rock_Wren.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumig" \
#     --image-path ./qual/original/Rock_Wren.png --label 197 --save-path ./qual/ig/Rock_Wren.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/Rock_Wren.png --label 197 --save-path ./qual/guidedpb/Rock_Wren.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "gradcam" \
#     --image-path ./qual/original/Rock_Wren.png --label 197 --save-path ./qual/gradcam/Rock_Wren.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "scorecam" \
#     --image-path ./qual/original/Rock_Wren.png --label 197 --save-path ./qual/scorecam/Rock_Wren.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/Rock_Wren.png --label 197 --save-path ./qual/ours/Rock_Wren.png --normalize --sign "positive"

# ### Geococcyx
# ### lrp
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumlrp" \
#     --image-path ./qual/original/Geococcyx.png --label 109 --save-path ./qual/lrp/Geococcyx.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumig" \
#     --image-path ./qual/original/Geococcyx.png --label 109 --save-path ./qual/ig/Geococcyx.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/Geococcyx.png --label 109 --save-path ./qual/guidedpb/Geococcyx.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "gradcam" \
#     --image-path ./qual/original/Geococcyx.png --label 109 --save-path ./qual/gradcam/Geococcyx.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "scorecam" \
#     --image-path ./qual/original/Geococcyx.png --label 109 --save-path ./qual/scorecam/Geococcyx.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/Geococcyx.png --label 109 --save-path ./qual/ours/Geococcyx.png --normalize --sign "positive"


# ### IMAGENET
# ### drumstick
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/lrp/drumstick.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/ig/drumstick.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/guidedpb/drumstick.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/gradcam/drumstick.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/scorecam/drumstick.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/ours/drumstick.png --normalize --sign "positive"
# ### ablation
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "simple" --heat-quantization \
#     --image-path ./qual/original/drumstick.png --label 542 --save-path ./qual/our-ablation/drumstick.png --normalize --sign "positive"

# ### bee
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/lrp/bee.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/ig/bee.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/guidedpb/bee.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/gradcam/bee.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/scorecam/bee.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/ours/bee.png --normalize --sign "positive"
# ### ablation
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "simple" --heat-quantization \
#     --image-path ./qual/original/bee.png --label 309 --save-path ./qual/our-ablation/bee.png --normalize --sign "positive"

# ### Arabian_camel
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/lrp/Arabian_camel.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/ig/Arabian_camel.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/guidedpb/Arabian_camel.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/gradcam/Arabian_camel.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/scorecam/Arabian_camel.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/ours/Arabian_camel.png --normalize --sign "positive"
# ### ablation
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "simple" --heat-quantization \
#     --image-path ./qual/original/Arabian_camel.png --label 354 --save-path ./qual/our-ablation/Arabian_camel.png --normalize --sign "positive"

# ### water_ouzel
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/lrp/water_ouzel.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/ig/water_ouzel.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/guidedpb/water_ouzel.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/gradcam/water_ouzel.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/scorecam/water_ouzel.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/ours/water_ouzel.png --normalize --sign "positive"
# ### ablation
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "simple" --heat-quantization \
#     --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/our-ablation/water_ouzel.png --normalize --sign "positive"

# ### ram
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/lrp/ram.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/ig/ram.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/guidedpb/ram.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/gradcam/ram.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/scorecam/ram.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/ours/ram.png --normalize --sign "positive"
# ### ablation
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "simple" --heat-quantization \
#     --image-path ./qual/original/ram.png --label 348 --save-path ./qual/our-ablation/ram.png --normalize --sign "positive"

# ### bustard
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/bustard.png --label 138 --save-path ./qual/lrp/bustard.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/bustard.png --label 138 --save-path ./qual/ig/bustard.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/bustard.png --label 138 --save-path ./qual/guidedpb/bustard.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/bustard.png --label 138 --save-path ./qual/gradcam/bustard.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/bustard.png --label 138 --save-path ./qual/scorecam/bustard.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/bustard.png --label 138 --save-path ./qual/ours/bustard.png --normalize --sign "positive"

# ### sock
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/sock.png --label 806 --save-path ./qual/lrp/sock.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/sock.png --label 806 --save-path ./qual/ig/sock.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/sock.png --label 806 --save-path ./qual/guidedpb/sock.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/sock.png --label 806 --save-path ./qual/gradcam/sock.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/sock.png --label 806 --save-path ./qual/scorecam/sock.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/sock.png --label 806 --save-path ./qual/ours/sock.png --normalize --sign "positive"

# ### wombat
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/wombat.png --label 106 --save-path ./qual/lrp/wombat.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/wombat.png --label 106 --save-path ./qual/ig/wombat.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/wombat.png --label 106 --save-path ./qual/guidedpb/wombat.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/wombat.png --label 106 --save-path ./qual/gradcam/wombat.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/wombat.png --label 106 --save-path ./qual/scorecam/wombat.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/wombat.png --label 106 --save-path ./qual/ours/wombat.png --normalize --sign "positive"


# ### Error Analysis
# ### bubble
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/bubble.png --label 971 --save-path ./qual/lrp/bubble.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/bubble.png --label 971 --save-path ./qual/ig/bubble.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/bubble.png --label 971 --save-path ./qual/guidedpb/bubble.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/bubble.png --label 971 --save-path ./qual/gradcam/bubble.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/bubble.png --label 971 --save-path ./qual/scorecam/bubble.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/bubble.png --label 971 --save-path ./qual/ours/bubble.png --normalize --sign "positive"

# ### solar_dish
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/solar_dish.png --label 807 --save-path ./qual/lrp/solar_dish.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/solar_dish.png --label 807 --save-path ./qual/ig/solar_dish.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/solar_dish.png --label 807 --save-path ./qual/guidedpb/solar_dish.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/solar_dish.png --label 807 --save-path ./qual/gradcam/solar_dish.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/solar_dish.png --label 807 --save-path ./qual/scorecam/solar_dish.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/solar_dish.png --label 807 --save-path ./qual/ours/solar_dish.png --normalize --sign "positive"

# ### oboe
# ### lrp
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumlrp" \
#     --image-path ./qual/original/oboe.png --label 683 --save-path ./qual/lrp/oboe.png --normalize --sign "positive"
# ### ig
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumig" \
#     --image-path ./qual/original/oboe.png --label 683 --save-path ./qual/ig/oboe.png --normalize --sign "positive"
# ### guidedpb
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "captumguidedbackprop" \
#     --image-path ./qual/original/oboe.png --label 683 --save-path ./qual/guidedpb/oboe.png --normalize --sign "positive"
# ### gradcam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "gradcam" \
#     --image-path ./qual/original/oboe.png --label 683 --save-path ./qual/gradcam/oboe.png --normalize --sign "positive"
# ### scorecam
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "scorecam" \
#     --image-path ./qual/original/oboe.png --label 683 --save-path ./qual/scorecam/oboe.png --normalize --sign "positive"
# ### ours
# poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
#     --skip-connection-prop-type "flows_skip" --heat-quantization \
#     --image-path ./qual/original/oboe.png --label 683 --save-path ./qual/ours/oboe.png --normalize --sign "positive"
