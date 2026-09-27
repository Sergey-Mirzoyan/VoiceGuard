import numpy as np
from scipy.stats import chi2
from sklearn.neural_network import MLPClassifier
from collections import Counter

def read_binary_string(content):
    """
    Parses binary string content.
    Args:
        content (str): The string content of the file.
    Returns:
        list[int]: List of bits, or None if invalid.
    """
    try:
        sequence = content.strip()
        if not all(bit in '01' for bit in sequence):
            return None
        return [int(bit) for bit in sequence]
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None

def analyze_sequence_direction(sequence, window_size, start_pos, direction='next'):
    """Analysis of sequence in one direction using Neural Network"""
    X_train = []
    y_train = []
    
    # Define training size as 20% of available data, max 50
    training_size = min(len(sequence) // 5, 50)
    
    # Form training set around start_pos
    if direction == 'next':
        train_range = range(max(0, start_pos - training_size//2), 
                          min(start_pos + training_size//2, len(sequence) - window_size))
        for i in train_range:
            X_train.append(sequence[i:i+window_size])
            y_train.append(sequence[i+window_size])
            
        # Predict sequence after start_pos
        test_range = range(start_pos, len(sequence) - window_size)
    else:
        train_range = range(min(len(sequence)-1, start_pos + training_size//2), 
                          max(window_size-1, start_pos - training_size//2), -1)
        for i in train_range:
            X_train.append(sequence[i-window_size:i])
            y_train.append(sequence[i-window_size-1])
            
        # Predict sequence before start_pos
        test_range = range(start_pos, window_size-1, -1)
            
    if len(X_train) == 0:
        return [], 0, 0
        
    # Create and train model
    clf = MLPClassifier(hidden_layer_sizes=(window_size,), max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    predictions = []
    correct_predictions = 0
    
    for i in test_range:
        if direction == 'next':
            window = sequence[i:i+window_size]
            actual = sequence[i+window_size]
            predicted = clf.predict([window])[0]
            pos = i+window_size
        else:
            window = sequence[i-window_size:i]
            actual = sequence[i-window_size-1]
            predicted = clf.predict([window])[0]
            pos = i-window_size-1
            
        predictions.append({
            'position': pos,
            'window': ''.join(map(str, window)),
            'predicted': int(predicted),
            'actual': int(actual),
            'correct': bool(predicted == actual)
        })
        correct_predictions += (predicted == actual)
    
    accuracy = correct_predictions / len(predictions) if predictions else 0
    return predictions, accuracy, len(predictions)

def statistical_prediction(sequence, window_size, start_pos, direction='next'):
    """Statistical analysis using Chi-Square method"""
    predictions = []
    total_chi_square = 0
    total_p_value = 0
    total_tests = 0
    
    # Define training size as 20% of available data, max 50
    training_size = min(len(sequence) // 5, 50)
    
    # Analyze frequencies in training set
    if direction == 'next':
        train_range = range(max(0, start_pos - training_size//2),
                          min(start_pos + training_size//2, len(sequence) - window_size))
        test_range = range(start_pos, len(sequence) - window_size)
    else:
        train_range = range(min(len(sequence)-1, start_pos + training_size//2),
                          max(window_size-1, start_pos - training_size//2), -1)
        test_range = range(start_pos, window_size-1, -1)
    
    # Collect statistics from training set
    window_stats = {}
    for i in train_range:
        if direction == 'next':
            window = tuple(sequence[i:i+window_size])
            next_bit = sequence[i+window_size]
        else:
            window = tuple(sequence[i-window_size:i])
            next_bit = sequence[i-window_size-1]
            
        if window not in window_stats:
            window_stats[window] = Counter()
        window_stats[window][next_bit] += 1
    
    # Make predictions
    for i in test_range:
        if direction == 'next':
            window = tuple(sequence[i:i+window_size])
            actual = sequence[i+window_size]
            pos = i+window_size
        else:
            window = tuple(sequence[i-window_size:i])
            actual = sequence[i-window_size-1]
            pos = i-window_size-1
            
        freq = window_stats.get(window, Counter())
        
        if sum(freq.values()) > 0:
            expected = sum(freq.values()) / 2
            chi_square = sum((obs - expected) ** 2 / expected for obs in freq.values())
            p_value = 1 - chi2.cdf(chi_square, df=1)
            
            predictions.append({
                'position': pos,
                'window': ''.join(map(str, window)),
                'actual': int(actual),
                'chi_square': float(chi_square),
                'p_value': float(p_value)
            })
            total_chi_square += chi_square
            total_p_value += p_value
            total_tests += 1
    
    avg_chi_square = total_chi_square / total_tests if total_tests > 0 else 0
    avg_p_value = total_p_value / total_tests if total_tests > 0 else 0
    return predictions, avg_chi_square, avg_p_value, total_tests

def get_cached_block_model(sequence, window_size):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import random
    
    valid_indices = list(range(len(sequence) - window_size))
    if len(valid_indices) > 50:
        indices = random.sample(valid_indices, 50)
    else:
        indices = valid_indices
        
    X_train, y_train = [], []
    for i in indices:
        X_train.append(sequence[i:i+window_size])
        y_train.append(sequence[i+window_size])
        
    if not X_train:
        return None, None
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = MLPRegressor(
        hidden_layer_sizes=(window_size*2, window_size),
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler

def block_predict_with_cached_model(sequence, position, window_size, predict_size, direction, model, scaler):
    if model is None or scaler is None:
        return None
        
    if direction == 'next':
        if position + window_size + predict_size > len(sequence):
            return None
        window = sequence[position:position+window_size]
        actual = sequence[position+window_size:position+window_size+predict_size]
    else:
        if position - window_size - predict_size < 0:
            return None
        window = sequence[position-window_size:position]
        actual = sequence[position-window_size-predict_size:position-window_size]
    
    predicted = []
    current = np.array(window)
    
    for _ in range(predict_size):
        current_scaled = scaler.transform(current.reshape(1, -1))
        pred = model.predict(current_scaled)[0]
        bit = 1 if pred >= 0.5 else 0
        predicted.append(bit)
        
        if direction == 'next':
            current = np.roll(current, -1)
            current[-1] = bit
        else:
            current = np.roll(current, 1)
            current[0] = bit
    
    matches = sum(a == b for a, b in zip(predicted, actual))
    accuracy = matches / predict_size if predict_size > 0 else 0
    
    return {
        'window': ''.join(map(str, window)),
        'predicted': ''.join(map(str, predicted)),
        'actual': ''.join(map(str, actual)),
        'accuracy': accuracy
    }

def get_block_transitions(sequence, window_size, predict_size):
    transitions = {}
    for i in range(len(sequence) - window_size - predict_size):
        current = tuple(sequence[i:i+window_size])
        next_block = tuple(sequence[i+window_size:i+window_size+predict_size])
        if current not in transitions:
            transitions[current] = {}
        if next_block not in transitions[current]:
            transitions[current][next_block] = 0
        transitions[current][next_block] += 1
    return transitions

def block_statistical_prediction_cached(sequence, position, window_size, predict_size, direction, transitions):
    from scipy.stats import chi2_contingency
    
    if direction == 'next':
        if position + window_size + predict_size > len(sequence):
            return None
        window = tuple(sequence[position:position+window_size])
        actual = tuple(sequence[position+window_size:position+window_size+predict_size])
    else:
        if position - window_size - predict_size < 0:
            return None
        window = tuple(sequence[position-window_size:position])
        actual = tuple(sequence[position-window_size-predict_size:position-window_size])
    
    if window not in transitions:
        return {
            'window': ''.join(map(str, window)),
            'predicted': '0' * predict_size,
            'actual': ''.join(map(str, actual)),
            'chi_square': 0,
            'p_value': 1.0
        }
    
    observed = np.zeros((2, 2))
    for next_block, count in transitions[window].items():
        row = 0 if next_block == actual else 1
        observed[row][0] += count
        observed[row][1] += sum(transitions[window].values()) - count
    
    chi2_val, p_value, _, _ = chi2_contingency(observed + 1)
    predicted = max(transitions[window].items(), key=lambda x: x[1])[0]
    
    return {
        'window': ''.join(map(str, window)),
        'predicted': ''.join(map(str, predicted)),
        'actual': ''.join(map(str, actual)),
        'chi_square': float(chi2_val),
        'p_value': float(p_value)
    }

def run_full_analysis(sequence, window_sizes=[8, 16, 32]):
    """
    Runs full analysis on the sequence.
    Returns a dictionary with structured results.
    """
    if not sequence:
        return None
    
    start_pos = len(sequence) // 2
    
    results = {
        'sequence_length': len(sequence),
        'bitwise': {
            'neural_network': {
                'details': [], 'overall_accuracy': 0, 'total_analyzed': 0, 'is_strong': False
            },
            'statistical': {
                'details': [], 'overall_chi_square': 0, 'overall_p_value': 0, 'total_tests': 0, 'is_strong': False
            }
        },
        'block': {
            'neural_network': {
                'details': [], 'overall_accuracy': 0, 'total_analyzed': 0, 'is_strong': False
            },
            'statistical': {
                'details': [], 'overall_chi_square': 0, 'overall_p_value': 0, 'total_tests': 0, 'is_strong': False
            }
        }
    }
    
    # === BITWISE ANALYSIS ===
    total_acc = 0
    total_an = 0
    for ws in window_sizes:
        for d in ['next', 'prev']:
            preds, acc, bits = analyze_sequence_direction(sequence, ws, start_pos, d)
            if bits > 0:
                results['bitwise']['neural_network']['details'].append({
                    'window_size': ws, 'direction': d, 'bits_analyzed': bits, 'accuracy': acc, 'predictions': preds
                })
                total_acc += acc * bits
                total_an += bits
    if total_an > 0:
        results['bitwise']['neural_network']['overall_accuracy'] = total_acc / total_an
        results['bitwise']['neural_network']['total_analyzed'] = total_an
        results['bitwise']['neural_network']['is_strong'] = (total_acc / total_an) <= 0.55

    tot_chi = 0
    tot_p = 0
    tot_t = 0
    for ws in window_sizes:
        for d in ['next', 'prev']:
            preds, avg_chi, avg_p, tests = statistical_prediction(sequence, ws, start_pos, d)
            if tests > 0:
                results['bitwise']['statistical']['details'].append({
                    'window_size': ws, 'direction': d, 'tests': tests, 'avg_chi_square': avg_chi, 'avg_p_value': avg_p, 'predictions': preds
                })
                tot_chi += avg_chi * tests
                tot_p += avg_p * tests
                tot_t += tests
    if tot_t > 0:
        results['bitwise']['statistical']['overall_chi_square'] = tot_chi / tot_t
        results['bitwise']['statistical']['overall_p_value'] = tot_p / tot_t
        results['bitwise']['statistical']['total_tests'] = tot_t
        results['bitwise']['statistical']['is_strong'] = (tot_p / tot_t) > 0.05

    # === BLOCKWISE ANALYSIS ===
    import random
    min_pos = 32
    max_pos = len(sequence) - 32
    if max_pos > min_pos:
        # Generate 50 random positions for thorough analysis
        positions = [random.randint(min_pos, max_pos) for _ in range(50)]
        
        block_nn_acc = 0
        block_nn_bits = 0
        
        block_stat_p = 0
        block_stat_count = 0
        
        for ws in window_sizes:
            # Cache the models and statistics ONCE per window size!
            model, scaler = get_cached_block_model(sequence, ws)
            transitions = get_block_transitions(sequence, ws, ws)
            
            for pos in positions:
                for d in ['next', 'prev']:
                    res_nn = block_predict_with_cached_model(sequence, pos, ws, ws, d, model, scaler)
                    if res_nn:
                        results['block']['neural_network']['details'].append({
                            'window_size': ws, 'direction': d, 'window': res_nn['window'], 
                            'predicted': res_nn['predicted'], 'actual': res_nn['actual'], 'accuracy': res_nn['accuracy']
                        })
                        block_nn_acc += res_nn['accuracy'] * ws
                        block_nn_bits += ws
                        
                    res_stat = block_statistical_prediction_cached(sequence, pos, ws, ws, d, transitions)
                    if res_stat:
                        results['block']['statistical']['details'].append({
                            'window_size': ws, 'direction': d, 'window': res_stat['window'],
                            'predicted': res_stat['predicted'], 'actual': res_stat['actual'],
                            'chi_square': res_stat['chi_square'], 'p_value': res_stat['p_value']
                        })
                        block_stat_p += res_stat['p_value']
                        block_stat_count += 1

        if block_nn_bits > 0:
            results['block']['neural_network']['overall_accuracy'] = block_nn_acc / block_nn_bits
            results['block']['neural_network']['total_analyzed'] = block_nn_bits
            results['block']['neural_network']['is_strong'] = (block_nn_acc / block_nn_bits) <= 0.55
            
        if block_stat_count > 0:
            results['block']['statistical']['overall_p_value'] = block_stat_p / block_stat_count
            results['block']['statistical']['total_tests'] = block_stat_count
            results['block']['statistical']['is_strong'] = (block_stat_p / block_stat_count) > 0.05

    return results
