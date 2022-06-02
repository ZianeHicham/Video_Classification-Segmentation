from flask import Flask,send_file
from flask_restful import Api, Resource
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation





app = Flask(__name__)
api = Api(app)
ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")

class classify(Resource):
    def get(self,video_path):
        x = ins.process_video("test/"+video_path+".avi", show_bboxes=True, frames_per_second=15,
         output_video_name="output_video.mp4") #output_video.mp4 stockée dans serveur 
        return send_file("output_video.mp4") #output_video.mp4 sera envoyer au client
        
api.add_resource(classify,"/check/<string:video_path>")


@app.route('/')
def index():
    return "Walcome to video classification api"

   
if __name__ == "__main__":
    
    app.run(debug = False)

