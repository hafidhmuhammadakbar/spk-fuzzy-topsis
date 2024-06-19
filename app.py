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
    
    # print(f"Number of decision makers: {num_dm}")

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

    # build the decision maker criteria
    dm_criteria = ft.build_dm_criteria(dm_criteria_input, num_dm)

    # build the aggregate decision maker criteria
    aggregated_criteria = ft.build_aggregate_dm_criteria(dm_criteria)

    # get the weighted normalized matrix
    weighted_normalized_matriks = ft.weighted_normalized_matrix(normalized_matrix, aggregated_criteria)

    # get the fpis and fnis
    fpis, fnis = ft.get_fpis_fnis(weighted_normalized_matriks, ft.variable_info())

    # get the distance from fpis and fnis
    distance_fpis, distance_fnis = ft.get_distance_from_fpis_fnis(weighted_normalized_matriks, fpis, fnis)

    # get the closest distance
    closest_distances = ft.get_closest_distance(distance_fpis, distance_fnis)

    # load the name alternatives
    name_alternatives = ft.name_alternatives()

    # load the name of criteria
    name_criterias = ft.name_criteria()

    # sort the alternatives based on the closest distance
    sorted_alternatives = sorted(range(len(closest_distances)), key=lambda k: closest_distances[k], reverse=True)

    return render_template('result-fuzzy.html', closest_distances=closest_distances, sorted_alternatives=sorted_alternatives, 
                        name_alternatives=name_alternatives, name_criterias=name_criterias, initial_matrix=data,
                        normalized_matrix=normalized_matrix, num_dm=num_dm, dm_initial_criteria=dm_criteria,
                        aggregated_criteria=aggregated_criteria, weighted_normalized_matriks=weighted_normalized_matriks,
                        fpis=fpis, fnis=fnis, distance_fpis=distance_fpis, distance_fnis=distance_fnis)

if __name__ == '__main__':
    app.run(debug=True)
