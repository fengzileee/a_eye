import sys
import time
sys.path.append("../../")
from appJar import gui
import simplenet
from aeye import *


app = gui()

def chooseFileName(btn):
    if btn == "Select Image From File":
        filename = app.openBox(title=None, dirName=None, fileTypes=None, asFile=False)
        app.reloadImage("reload", filename)

        img = mimg.imread(filename)
        shape = np.shape(img)
        ratio = float(shape[1])/float(shape[0])
        print(ratio)

        if shape[1] > 250 and ratio > 1.3 and ratio < 1.7:
            result = predict(filename, predict_operation, X_ph,
                    keep_p_ph, sess)
            app.setLabel("results", 
                    "You have {:3f} percent chance of getting cancer" \
                            .format(result[1]*100))
        else:
            app.setLabel("results", 
                    "ERRORRRRRRRRRRRRRRRRRRR!!!!!!! \n" \
                    + "Image too small or image's ratio too strange. "   \
                    + "Recommend around 460 x 700")


# ============== loading model =================
split_index = 3
model_name = 'model_cnn_' + str(split_index) + '.ckpt'
model_dir = './model'
model_file = model_dir + '/' + model_name

X_ph, Y_ph, keep_p_ph, training_operation, accuracy_operation,\
        predict_operation, out\
        = construct_graph(network_builder = simplenet.conv_net, 
                learning_rate = 0.00001)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_file)


# ============== setting up gui ================

app.setGeometry("1200x500")
app.addButton("Select Image From File", chooseFileName,0,0)
app.setBg("white")
app.setResizable(canResize=True)



app.startLabelFrame("Results")
app.addLabel("results","Please Begin by loading a image",1,0, colspan=3)
app.stopLabelFrame()

#Frame containing original "appear here" picture #
app.startLabelFrame("Image", 0,1, rowspan = 2)
app.setSticky("ew")
app.addImage("reload", "./cover.png")
app.zoomImage("reload", -2)
app.stopLabelFrame()

app.go()



#For Drag and Drop into textbox #
##app.addLabel("Label", text="Drop your file in the blank space below:",row=0,column=0)
##app.addEntry("dnd_file",1,0)
##app.setEntryDropTarget("dnd_file", function=None, replace=True)


##app.addButton("Load Image", changePic,3,0)





