
import re
import string

TEMPLATE_OBJCOUNTS = string.Template("""Instruction: Given an image caption, determine the objects and its counts to draw an image.                                               
Caption: $PROMPT""")

TEMPLATE_OBJCOORDS = string.Template("""Instruction: Given an image caption and objects, determine the coordinates of the objects.
Caption: $PROMPT
Objects: $OBJECTS""")


def normalize_quantize_coordinates(box, width, height, n_bins=100, normalize=True, quantize=True, target_format='xyxy'):
    """
    box: [x1, y1, x2, y2], # unnormalized box coordinates
    """

    x1, y1, x2, y2 = box

    # normalized
    if normalize:
        x1 = x1 / width
        y1 = y1 / height
        x2 = x2 / width
        y2 = y2 / height

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(1, x2)
        y2 = min(1, y2)

    # discretized into n_bins
    # 0 ~ n_bins-1
    if quantize:
        # x1, y1, x2, y2 = [np.digitize([coord], get_bins(n_bins))[0] for coord in [x1, y1, x2, y2]]
        # x1, y1, x2, y2 = np.digitize([x1, y1, x2, y2], get_bins(n_bins)).tolist()

        num_bins = n_bins

        # quant_x0 = int(round((box[0] * (num_bins - 1))))
        # quant_y0 = int(round((box[1] * (num_bins - 1))))
        # quant_x1 = int(round((box[2] * (num_bins - 1))))
        # quant_y1 = int(round((box[3] * (num_bins - 1))))

        x1 = int(round((x1 * (num_bins - 1))))
        y1 = int(round((y1 * (num_bins - 1))))
        x2 = int(round((x2 * (num_bins - 1))))
        y2 = int(round((y2 * (num_bins - 1))))

    assert target_format in ['xyxy', 'xywh', 'cxcywh'], f"target_format: {target_format}"

    if target_format == 'xyxy':
        box = [x1, y1, x2, y2]

    elif target_format == 'xywh':
        w, h = x2 - x1, y2 - y1
        w = max(0, w)
        h = max(0, h)
        box = [x1, y1, w, h]

    elif target_format == 'cxcywh':
        w, h = x2 - x1, y2 - y1
        w = max(0, w)
        h = max(0, h)
        cx = x1 + w/2
        cy = y1 + h/2
        box = [cx, cy, w, h]
    
    return box


def prepare_task(caption, objects, task='predict_box_captions', mask_all_objects=True, dataset='flickr30k'):
    """
    caption: "A man with a red helmet on a small moped on a dirt road."
    
    objects: list of boxes
        box = {
            'text': box_caption,
            'box': [x1, y1, x2, y2], # normalized/quantized box coordinates
        }

    
    """

    tasks = ['predict_box_captions', 'predict_box_coordinates']
    assert task in tasks, f"task must be one of {tasks}, but got {task}"

    if dataset == 'flickr30k':
            
        if task == 'predict_box_captions':
            instruction = 'Given an image caption, determine objects and their counts.'
            out = prepare_task_predict_box_captions_flickr(caption, objects, instruction=instruction)

        if task == 'predict_box_coordinates':
            instruction = 'Given an image caption and objects, determine the coordinates of the objects.'
            out = prepare_task_predict_box_coordinates_flickr(caption, objects, instruction=instruction)

    assert 'source_text' in out, f"source_text must be in the output, but got {out.keys()}"
    assert 'target_text' in out, f"target_text must be in the output, but got {out.keys()}"

    return out


# Given an image caption, determine the objects and its counts to draw an image.
# Example Caption: Two skiers, one in pink and one in black, ride on a ski lift.
# Example output: Two skiers (2), pink (1), black (1), a ski lift (1)

def prepare_task_predict_box_captions_flickr(
        caption,
        objects,
        # mask_all_objects=False,
        instruction='Given an image caption, determine the objects and its counts to draw an image.'):
    """
    objects: list of boxes
        box = {
            'text': box_caption,            
        }

    INPUT:
        Instruction: Given an image caption, determine the objects and its counts to draw an image.
        Caption: Two skiers, one in pink and one in black, ride on a ski lift.

    OUTPUT:
        Two skiers (2), pink (1), black (1), a ski lift (1)
    """

    # {'image_id': '3211453055',
    #  'width': 332,
    #  'height': 500,
    #  'depth': 3,
    #  'sentence': 'Two skiers , one in pink and one in black , ride on a ski lift .',
    #  'regions': [{'id': '101454',
    #               'text': 'Two skiers',
    #               'phrase_type': ['people'],
    #               'first_word_index': 0,
    #               'boxes': [(124, 168, 290, 359), (1, 188, 134, 347)]},
    #              {'id': '101456',
    #               'text': 'pink',
    #               'phrase_type': ['clothing'],
    #               'first_word_index': 5,
    #               'boxes': [(0, 229, 130, 325)]},
    #              {'id': '101458',
    #               'text': 'black',
    #               'phrase_type': ['clothing'],
    #               'first_word_index': 9,
    #               'boxes': [(117, 217, 268, 325)]},
    #              {'id': '101459',
    #               'text': 'a ski lift',
    #               'phrase_type': ['scene'],
    #               'first_word_index': 13,
    #               'boxes': [(3, 230, 203, 417)]}]}

    # objects = regions

    # prepare input
    input_text = []
    # f"Instruction: {instruction} \n Caption: {caption}"
    input_text.append(f"Instruction: {instruction}")
    input_text.append(f"Caption: {caption}")
    input_text = '\n'.join(input_text)

    # prepare output
    output_text = []
    for obj in objects:
        n_boxes = len(obj['boxes'])
        text = obj['text']
        output_text.append(f"{text} ({n_boxes})")
    output_text = ', '.join(output_text)

    return {
        'source_text': input_text,
        'target_text': output_text,

        'target_objects': objects,
    }

def decode_objects_from_text(input_string):
    # Regular expression to match words and counts
    pattern = r"((?:[\w'-]+\s+)*[\w'-]+)\s+\((\d+)\)"
    matches = re.findall(pattern, input_string)
    
    result = []
    for match in matches:
        words, count = match
        # Remove any leading/trailing whitespace from the words
        words = words.strip()
        # Convert the count string to an integer
        count = int(count)
        # Append the words and count to the result list
        result.append({"text": words, "count": count})
    return result

print('Example Object parsing')
input_string = "the word 'START' (1) a blue t-shirt (1)"
result = decode_objects_from_text(input_string)
print(result)

input_string = "Two skiers (2), pink (1), black (1), a ski lift (1)"
result = decode_objects_from_text(input_string)
print(result)
# [{'text': 'Two skiers', 'count': 2}, {'text': 'pink', 'count': 1}, {'text': 'black', 'count': 1}, {'text': 'a ski lift', 'count': 1}]


def prepare_task_predict_box_coordinates_flickr(caption,
                                      objects,
                                    #   mask_all_objects=False,
                                      instruction='Given an image caption and objects, determine the coordinates of the objects'):
    """
    objects: list of boxes
        box = {
            'text': box_caption,
            
        }

    Example 
        
    INPUT:
        Instruction: Given an image caption and objects, determine the coordinates of the objects.
        Caption: Two skiers, one in pink and one in black, ride on a ski lift.
        Objects: Two skiers (2), pink (1), black (1), a ski lift (1)
        
    OUTPUT:
        Two skiers [(124, 168, 290, 359), (1, 188, 134, 347)] pink [(0, 229, 130, 325)] black [(117, 217, 268, 325)] a ski lift [(3, 230, 203, 417)]
    """


    object_text = []
    for obj in objects:
        n_boxes = len(obj['boxes'])
        text = obj['text']
        object_text.append(f"{text} ({n_boxes})")
    object_text = ', '.join(object_text)

    input_text = []
    input_text.append(f"Instruction: {instruction}")
    input_text.append(f"Caption: {caption}")
    input_text.append(f"Objects: {object_text}")
    input_text = '\n'.join(input_text)

    # prepare output

    output_text = []
    for obj in objects:
        
        # Two skiers
        text = obj['text']

        # [(124, 168, 290, 359), (1, 188, 134, 347)]
        boxes = obj['boxes']  # list of tuples

        obj_text = []

        obj_text += [text]

        box_coordinates = []
        for box in boxes:
            box_coordinates += [f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"]
        obj_text += [
            "[" +  ", ".join(box_coordinates) + "]"
        ]

        obj_text = " ".join(obj_text)
        output_text.append(obj_text)

    output_text = " ".join(output_text)

    return {
        'source_text': input_text,
        'target_text': output_text,

        'target_objects': objects,
    }

def decode_coordinates_from_text(input_string):
    # Regular expression to match multiple words and coordinate tuples
    pattern = r"((?:[\w'-]+\s+)*[\w'-]+)\s+\[([^\]]+)\]"
    matches = re.findall(pattern, input_string)
    
    result = []
    for match in matches:
        words, coordinates_str = match
        # Remove any leading/trailing whitespace from the words
        words = words.strip()
        # Regular expression to match individual tuples within the coordinates string
        coord_pattern = r"\(([^)]+)\)"
        coord_matches = re.findall(coord_pattern, coordinates_str)
        # Convert the string representation of each tuple into an actual tuple
        coordinates = [tuple(map(int, coord_str.split(','))) for coord_str in coord_matches]
        # Append the words and coordinates to the result list
        result.append({"text": words, "boxes": coordinates})
    return result


# Example usage
input_string = "Two skiers [(124, 168, 290, 359), (1, 188, 134, 347)] pink [(0, 229, 130, 325)] black [(117, 217, 268, 325)] a ski lift [(3, 230, 203, 417)]"
parsed_data = decode_coordinates_from_text(input_string)
print(parsed_data)

input_string = "A man [(7, 8, 11, 19)] a reflective vest [(7, 8, 11, 19)] a hard hat [(7, 8, 11, 19)] a flag [(7, 8, 11, 19)]"
print(decode_coordinates_from_text(input_string))

input_string = "the word 'START' [(7, 8, 11, 19)] a reflective vest [(7, 8, 11, 19)]"
print(decode_coordinates_from_text(input_string))
