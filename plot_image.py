import cv2 as cv
def plot_image(image,boxes):
    img = cv.imread(image)

# box = [class_prob, IOU_prob,x,y,w,h]
# box_format is 'mid-point'
# boxes[0]=x, box[1]=y (center)
# boxes[2]=width, boxes[3]=height
    for box in boxes:
        box = box[2:]
        assert len(box) == 4
        x1 = int(box[0] - (box[2]/2))
        y1 = int(box[1] - (box[3]/2))
        x2 = int(box[0] + (box[2]/2))
        y2 = int(box[1] + (box[3]/2))

# cv.retangle(image,(x1,y1),(x2,y2),color)
#     cv.imshow('image',img)
        rect = cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),thickness=2)
    cv.imshow('image',img)
    cv.waitKey(0)

# testing:
# image = 'img2.jpeg'
# boxes = [[22,23,300,400,100,100]]
# plot_image(image,boxes)


