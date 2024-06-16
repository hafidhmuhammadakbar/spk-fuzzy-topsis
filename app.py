from flask import Flask, render_template, request
import numpy as np
from src import fuzzy_topsis as ft

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fuzzy', methods=['GET'])
def fuzzy():
    return render_template('fuzzy.html')

@app.route('/fuzzy', methods=['POST'])
def fuzzy_post():
    # Log the entire form data for debugging
    app.logger.info(request.form)

    # Determine the number of decision makers based on submitted data
    num_dm = 0
    criteria_name_length = len(ft.name_criteria())
    
    for key, value in request.form.items():
        if key.startswith('decisionMaker') and key.endswith('Criteria1'):
            num_dm += 1
    
    print(f"Number of decision makers: {num_dm}")

    if num_dm <= 0:
        return "Error: Invalid number of decision makers submitted", 400

    dm_criteria_input = np.zeros((num_dm, criteria_name_length))

    for i in range(num_dm):
        for j in range(criteria_name_length):
            dm_criteria_input[i][j] = int(request.form.get(f'decisionMaker{i+1}Criteria{j+1}', 0))

    # Load the data
    data = ft.load_variable()

    # Get the normalized matrix
    normalized_matrix = ft.get_normalize(data)

    dm_criteria = ft.build_dm_criteria(dm_criteria_input, num_dm)

    aggregated_criteria = ft.build_aggregate_dm_criteria(dm_criteria)

    weighted_normalized_matriks = ft.weighted_normalized_matrix(normalized_matrix, aggregated_criteria)

    fpis, fnis = ft.get_fpis_fnis(weighted_normalized_matriks, ft.variable_info())

    distance_fpis, distance_fnis = ft.get_distance_from_fpis_fnis(weighted_normalized_matriks, fpis, fnis)

    closest_distance = ft.get_closest_distance(distance_fpis, distance_fnis)

    name_alternatives = ft.name_alternatives()

    sorted_alternatives = sorted(range(len(closest_distance)), key=lambda k: closest_distance[k], reverse=True)

    length = len(closest_distance)

    return render_template('result-fuzzy.html', closest_distance=closest_distance, sorted_alternatives=sorted_alternatives, name_alternatives=name_alternatives, length=length)

if __name__ == '__main__':
    app.run(debug=True)
