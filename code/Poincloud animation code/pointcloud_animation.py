import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


folder_path = "data/pointclouds/*.xyz"

xyz_files = glob.glob(folder_path)

# create an empty list to hold numpy arrays
arrays_list = []

# loop through each file and read its contents into a numpy array
n = 0

for file_path in xyz_files:
    with open(file_path, 'r') as file:
        file_contents = file.readlines()
        file_array = np.array([np.array(line.strip().split()).astype(float) for line in file_contents])
        arrays_list.append(file_array)

# convert the list of numpy arrays into a numpy array of arrays
numpy_array = np.array(arrays_list, dtype=object)

number_of_files = len(numpy_array)

number_of_sets = 20
number_of_rows = 5
number_of_frames = 30
n = 0

for i in range(number_of_files):
    # extract x, y, and z values from the array
    x = numpy_array[i][:, 0]
    y = numpy_array[i][:, 1]
    z = numpy_array[i][:, 2]



for set in range(number_of_sets):
    set_id = str(set).zfill(3)
    for i in range(number_of_frames):
        angle = 360 / number_of_frames * i
        id = str(i).zfill(5)
        for row in range(number_of_rows):
            #print(f"set: {set}, row: {row}")

            # building_n = set * number_of_rows + row
            # car_n = set * number_of_rows + row + 100
            # fence_n = set * number_of_rows + row + 200
            # pole_n = set * number_of_rows + row + 300
            # tree_n = set * number_of_rows + row + 400

            #output singlar images
            for column in range(5):

                object_n = set * number_of_rows + row + column * 100
                object = numpy_array[object_n]

                x = object[:, 0]
                y = object[:, 1]
                z = object[:, 2]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_axis_off()
                ax.view_init(elev=30, azim=0 + (angle))

                if object_n < 100:
                    ax.scatter(x, y, z, c='firebrick', marker='o', s = 8)
                elif object_n < 200:
                    ax.scatter(x, y, z, c='navy', marker='o', s = 8)
                elif object_n < 300:
                    ax.scatter(x, y, z, c='grey', marker='o', s = 8)
                elif object_n < 400:
                    ax.scatter(x, y, z, c='orchid', marker='o', s = 8)
                elif object_n < 500:
                    ax.scatter(x, y, z, c='seagreen', marker='o', s = 8)



                plt.savefig(f"pointclouds/{set_id}_{id}_{angle}_{row}_{column}.png")
                plt.close(fig)



        tile_size = 150
        grid_size = (5, 5)
        images = glob.glob("pointclouds/*.png")
        images_per_image = 5 * number_of_rows
        images_to_skip = images_per_image*number_of_frames*set + i*images_per_image

        #print("*ANGLE*", angle)

        #print ("\tSet: ", set)
        #print("\tFrame n:", i)
        #print("\tImages per image:", images_per_image)
        #print("\tNumber of frames n:", number_of_frames)
        #print("\tImages to skip: ", images_to_skip)
        #print("\tNumber of images:", len(images))
        #print(images)

        image_sets = []


        j = images_to_skip

        image_sets.append(images[j:j + 25])


        # for j in range(0+images_to_skip, len(images)-images_to_skip, 25):
        #     print("\t\tj = ", j)
        #     print("\t\tIMAGEs: ", images[j:j+25])
        #     image_sets.append(images[j:j + 25])

        #print("j = ", j)
        #print("First image:", images[j])
        #print("IMAGES: ", images)
        #print("IMAGE SETS: ",image_sets)
        for h, images in enumerate(image_sets):
            # Create a new image to hold the tiled images
            result = Image.new('RGB', (grid_size[0] * tile_size, grid_size[1] * tile_size))

            # Iterate over each image and paste it into the final image
            for k in range(len(images)):

                x = k % grid_size[0]
                y = k // grid_size[0]
                tile = Image.open(images[k])
                tile = tile.resize((tile_size, tile_size))
                result.paste(tile, (x * tile_size, y * tile_size))

            result.save(f'pointclouds/sets/{str(n).zfill(3)}.png')
            n += 1
        #print("-------------------------------")
            # Save the final image






