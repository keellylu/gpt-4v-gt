from overeasy import *
from PIL import Image

workflow = Workflow([
    BoundingBoxSelectAgent(classes=["CLASS_NAME"], model=OwlV2()),
    NMSAgent(iou_threshold=0.5, score_threshold=0),
])

image_path = "PATH"
image = Image.open(image_path)
result, graph = workflow.execute(image)

workflow.visualize(graph)
