import random
def should_allocate_to_evaluation(new_records,new_records_ids, evaluation_start_threshold=250, evaluation_percentage=20):
    allocation_list = []
    for i in range(len(new_records)):
        if new_records_ids[i] < evaluation_start_threshold:
            allocation_list.append(0)
        else:
            allocation_list.append(1 if random.randint(1, 100) <= evaluation_percentage else 0)
    return allocation_list
