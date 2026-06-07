from Code.Data.DataIO import load_data

X_train, Y_train = load_data(processed_data_path="Data/Solver/ProcessedPlain", encode=False, output_type="move", include_prev_moves_input=False)
print(type(X_train))
print(X_train[0])
