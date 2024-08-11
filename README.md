Shape Processing and Symmetry Detection
This application allows you to upload a CSV file containing shape data, process the shapes, detect symmetry, and visualize the results. It also provides options to download the processed shapes in PNG or SVG format.

Video Link:https://vimeo.com/997356269?share=copy

Features
Upload CSV File: Allows you to upload a CSV file containing shape data.
Shape Processing: Corrects and approximates shapes based on their geometry.
Symmetry Detection: Detects symmetry lines for circles and polygons.
Visualization: Displays processed shapes with or without symmetry lines.
Download Options: Provides options to download processed shapes as PNG or SVG files.
Requirements
To run this application, you'll need to have the following Python libraries installed:

streamlit
numpy
matplotlib
opencv-python
svgwrite
You can install these dependencies using pip:

bash
Copy code
pip install streamlit numpy matplotlib opencv-python svgwrite
How to Use
Run the Application

Save the code to a file, e.g., app.py, and run it using the following command:

bash
Copy code
streamlit run app.py
Upload CSV File

Click on the "Upload CSV File" button to upload a CSV file with shape data.
The CSV file should be in the format where each row represents a point with the following columns: path_id, x, y.
Process Shapes

You can choose to display symmetry lines by checking the "Show Symmetry Lines" checkbox.
The application will process the shapes and display the results.
Download Processed Shapes

After processing, you will see options to download each shape as a PNG or SVG file.
Click on the "Download" buttons to save the files to your local machine.
CSV File Format
The CSV file should have the following format:

python
Copy code
path_id,x,y
0,1.0,1.0
0,2.0,2.0
...
path_id identifies different shapes or paths.
x and y represent the coordinates of the points.
Functions
read_csv(csv_path): Reads a CSV file and extracts shape data.
fit_circle(points): Fits a circle to a set of points.
is_near_circle(points, threshold=0.1): Checks if a set of points is near a circle.
find_symmetry_lines_for_circle(center, radius, n=6): Finds symmetry lines for a circle.
find_symmetry_lines_for_polygon(points): Finds symmetry lines for a polygon.
plot_shapes_only(ax, shapes, xlim, ylim): Plots shapes without symmetry lines.
plot_shapes_and_symmetry(ax, shapes, xlim, ylim): Plots shapes with symmetry lines.
plot_initial(paths_XYs): Plots the initial shapes from the CSV file.
plot_symmetry_lines(ax, symmetry_lines): Plots symmetry lines.
correct_shape(points, scale_factor=0.5): Corrects and approximates shapes.
draw_shapes_on_canvas(shapes, canvas_size=(1000, 1000)): Draws shapes on a canvas.
detect_and_fill_occlusions(canvas, original_shapes, canvas_size=(1000, 1000)): Detects and fills occlusions in shapes.
save_shape_as_png(shape, buffer, canvas_size=(1000, 1000)): Saves a shape as a PNG file.
save_shape_as_svg(shape, buffer, canvas_size=(1000, 1000)): Saves a shape as an SVG file.
streamlit_interface(): Defines the Streamlit application interface.
