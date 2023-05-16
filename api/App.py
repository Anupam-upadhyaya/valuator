from flask import Flask, request, jsonify
from newtrain import  predict_random_forest
import numpy as np
import json

app = Flask(__name__)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/api', methods=['GET'])
def calculate():
    # Get the input values from the request
    # input_values = request.json['input_values']
    query= request.args.get('data')
    if query is None:
     return jsonify({'error':'no data'})
    try:
     query= request.args.get('data')
     jsondata =json.loads(query)
     data2 = np.array(jsondata)
     datap = data2.tolist()
    #  data_list = data.tolist()
     print(data2)
    except ValueError:
     return jsonify({'error':'problem with data'})
     
    output = predict_random_forest(datap)
    response = {
        'output': output
    }

    print(query)

      

    # input_X = int(request.args(['query']))
    # new_X = request.json['input_values']

    # Perform some calculation using the input values
    # output = train_random_forest()

    

    # Create a dictionary to hold the output
    

    # Return the output as JSON
    # return jsonify(response)
    return json.dumps(response, cls=NumpyEncoder)

if __name__ == '__main__':
    app.run()
