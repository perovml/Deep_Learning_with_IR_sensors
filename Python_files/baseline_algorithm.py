def print_scene(scene):
    scene = scene.reshape(scene.shape[-1],scene.shape[-2])
    fig, ax = plt.subplots()
    im = ax.imshow(scene)
    fig.tight_layout()
    #print(f'Predicted: {predicted}, Real: {label}')
    for (j,i),temp in np.ndenumerate(scene):
        ax.text(i,j, round(temp, 1),ha='center',va='center', fontsize=8)
    plt.show()

def print_scene_labeled(scene, object_centers):
    scene = scene.reshape(scene.shape[-1], scene.shape[-2])
    fig, ax = plt.subplots()
    im = ax.imshow(scene)
    fig.tight_layout()
    
    # Plotting the centers of the objects
    for center in object_centers:
        ax.plot(center[1], center[0], 'o', color='darkblue')  # Object centers are marked in dark blue

    # Adding temperature labels as before
    for (j, i), temp in np.ndenumerate(scene):
        ax.text(i, j, round(temp, 1), ha='center', va='center', fontsize=8)
    plt.show()


def adjust_delta_by_skewness(arr, base_delta, skewness_scale_factor):
    """
    Adjusts the delta value based on the skewness of the data.

    :param arr: The input array (e.g., temperature data).
    :param base_delta: The base delta value.
    :param skewness_scale_factor: A factor to scale the adjustment of delta.
    :return: Adjusted delta value.
    """
    # Calculate skewness
    data_skewness = skew(arr.flatten())

    # Adjust delta based on skewness
    if data_skewness > 1:
        # Decrease delta for positive skewness
        adjusted_delta = base_delta * (1 - min(data_skewness * 1.7*skewness_scale_factor, 0.7))
    elif data_skewness < - 0.5:
        # Increase delta for small or negative skewness
        adjusted_delta = base_delta * (1 + min(abs(data_skewness) * (skewness_scale_factor), 0.1))
    else:
        adjusted_delta = base_delta    
    return adjusted_delta


def detect_local_maxima(arr, delta, mode_delta, check_hood, adj_delta = (False, 0.2)):
    # Apply padding to the array with edge values
    padded_arr = np.pad(arr, pad_width=1, mode='edge')
    #adjust_delta_based_on_skewness
    if adj_delta[0]:
        delta = adjust_delta_by_skewness(arr, delta, adj_delta[1])
    # Generate a structuring element that defines the neighborhood
    neighborhood = generate_binary_structure(2, 1)
    
    # Apply a maximum filter directly to the padded array (without smoothing)
    local_max = maximum_filter(padded_arr, footprint=neighborhood) == padded_arr
    average_value = 0
    # Calculate the average value of the array
    if mode_delta == 'std':
        std = np.std(arr)
        average_value = np.mean(arr)
        delta = delta*std
    if mode_delta == 'percentile':
        delta = delta*np.percentile(arr, 20)    
    if mode_delta == 'max':
        average_value = np.max(arr)
        delta = - delta
    #mode_value = stats.mode(arr, axis=None)[0]
    # Create a boolean array for detected peaks within the original array dimensions
    detected_peaks = np.zeros_like(arr, dtype=bool)
    coord_list = []

    # Iterate through the original array dimensions to find local maxima
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Adjust indices to account for padding:
            ip, jp = i + 1, j + 1
            if local_max[ip, jp]:
                neighborhood_values = padded_arr[ip-1:ip+2, jp-1:jp+2]
                # Check if the current point is a true local maximum and greater than the threshold value
                if (padded_arr[ip, jp] == np.max(neighborhood_values) and padded_arr[ip, jp] > average_value + delta):
                    if check_hood:
                        if np.sum(neighborhood_values > average_value + delta) >= 2:  # At least 2 pixels are above background threshold
                                detected_peaks[i, j] = True
                                coord_list.append((i, j))
                    else: 
                        detected_peaks[i, j] = True
                        coord_list.append((i,j))

    return coord_list

def refine_peaks_by_neighborhood_sum(coord_list, arr):
    refined_coords = []
    removed_indices = set()
    padded_arr = np.pad(arr, pad_width=1, mode='edge')

    for i, (x1, y1) in enumerate(coord_list):
        if i in removed_indices:
            continue

        # Adjust coordinates for the padded array
        x1_padded, y1_padded = x1 + 1, y1 + 1

        max_sum = np.sum(padded_arr[x1_padded:x1_padded+3, y1_padded:y1_padded+3])
        max_index = i

        # Check for neighboring peaks
        for j, (x2, y2) in enumerate(coord_list):
            if i != j and abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                # Adjust coordinates for the padded array
                x2_padded, y2_padded = x2 + 1, y2 + 1

                # Calculate neighborhood sum
                current_sum = np.sum(padded_arr[x2_padded:x2_padded+3, y2_padded:y2_padded+3])
                if current_sum > max_sum:
                    max_sum = current_sum
                    removed_indices.add(max_index)
                    max_index = j
                else:
                    removed_indices.add(j)

        if max_index not in removed_indices:
            refined_coords.append(coord_list[max_index])

    return refined_coords

def get_coordinates(img, delta, refine, mode_delta, check_hood, adj_delta):
    arr = img.reshape(img.shape[-1],img.shape[-2])
    coord_list = detect_local_maxima(arr, delta, mode_delta, check_hood, adj_delta)
    if refine:
        coord_list = refine_peaks_by_neighborhood_sum(coord_list, arr)
    return coord_list

def calculate_accuracy(img_list, label, delta, refine = True, 
                       mode_delta = 'percentile', check_hood = False, adj_delta = (False, 0.2)):
    peak_coord_list = []
    len_coord_list = []
    misclassified = []
    correctly_classified = []
    #skewness_list = []
    #weird_small, weird_big = [], []
    for i, img in enumerate(img_list):
        #skewness = skew(img.flatten())
        #delta = delta/skewness
        #skewness_list.append(skewness)
        coords = get_coordinates(img, delta, refine, mode_delta, check_hood, adj_delta)
        peak_coord_list.append(coords)
        #if skewness < 0:
        #    weird_small.append(i)
        #if skewness > 1.2:
        #    weird_big.append(i)
        if len(coords) == label:
            correctly_classified.append(i)
        if len(coords) != label:
            misclassified.append(i)
    accuracy = 1 - (len(misclassified)/len(img_list))
    print(f'accuracy: {accuracy}')
  # print(f'mean skewness: {np.mean(skewness_list)}')
  #  print(f'max skewness: {np.max(skewness_list)}')
  #  print(f'std skewness: {np.std(skewness_list)}')
  #  print(f'min skewness: {np.min(skewness_list)}')
    return peak_coord_list, misclassified, correctly_classified
    

def create_binary_image(peak_coord_list, size=(8, 8)):
    # Create an empty binary image
    binary_image = np.zeros(size, dtype=np.uint8)

    # Mark the peaks in the binary image
    for x, y in peak_coord_list:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            binary_image[x, y] = 1  # y corresponds to row and x to column

    return binary_image



def create_binary_image_fake(label, size = (8, 8)):
    # Create an empty binary image
    binary_image = np.zeros(size, dtype=np.uint8)

    # Ensure label does not exceed the total number of pixels in the image
    num_pixels = size[0] * size[1]
    assert label <= num_pixels, "label value exceeds the total number of pixels"

    # Randomly select positions to set to 1
    # np.random.choice generates a flat array of unique positions, then unravel_index converts them to 2D positions
    indices = np.random.choice(num_pixels, label, replace=False)
    x, y = np.unravel_index(indices, size)

    # Mark the selected positions in the binary image
    binary_image[x, y] = 1

    return binary_image