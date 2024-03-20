import random
def should_allocate_to_evaluation(new_records, current_total_records, evaluation_start_threshold=50, evaluation_percentage=20):
    allocation_list = []
    for i in range(new_records):
        # Decide allocation based on whether the current_total_records is below the threshold
        if current_total_records < evaluation_start_threshold:
            allocation_list.append(0)
        else:
            allocation_list.append(1 if random.randint(1, 100) <= evaluation_percentage else 0)
        current_total_records += 1
    return allocation_list
