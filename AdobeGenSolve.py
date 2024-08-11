import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import svgwrite
from io import BytesIO

# Read CSV file
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Fit a circle to points
def fit_circle(points):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.mean(distances)
    return center, radius

# Check if points are near a circle
def is_near_circle(points, threshold=0.1):
    center, radius = fit_circle(points)
    distances = np.linalg.norm(points - center, axis=1)
    avg_distance = np.mean(distances)
    distance_variation = np.std(distances) / avg_distance
    return distance_variation < threshold

# Find symmetry lines for circles
def find_symmetry_lines_for_circle(center, radius, n=6):
    symmetry_lines = []
    angles = np.linspace(0, np.pi, n, endpoint=False)
    for angle in angles:
        theta = np.array([np.cos(angle), np.sin(angle)])
        point1 = center + radius * theta
        point2 = center - radius * theta
        line = np.array([point1, point2])
        symmetry_lines.append(line)
    return symmetry_lines

# Find symmetry lines for polygons
def find_symmetry_lines_for_polygon(points):
    symmetry_lines = []
    centroid = np.mean(points, axis=0)
    num_vertices = len(points)
    side_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)

    if num_vertices % 2 == 0 and np.allclose(side_lengths[:num_vertices//2], side_lengths[num_vertices//2:]):
        midpoints = find_midpoints_of_opposite_sides(points)
        for midpoint1, midpoint2 in midpoints:
            line = np.array([midpoint1, midpoint2])
            symmetry_lines.append(line)
        
        if len(points) == 4:
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            vertical_symmetry = np.array([[np.mean(x_coords), np.min(y_coords)],
                                          [np.mean(x_coords), np.max(y_coords)]])
            horizontal_symmetry = np.array([[np.min(x_coords), np.mean(y_coords)],
                                            [np.max(x_coords), np.mean(y_coords)]])
            symmetry_lines.append(vertical_symmetry)
            symmetry_lines.append(horizontal_symmetry)
    else:
        for i in range(num_vertices // 2):
            vertex1 = points[i]
            vertex2 = points[(i + num_vertices // 2) % num_vertices]
            line = np.array([vertex1, vertex2])
            symmetry_lines.append(line)
    
    return symmetry_lines

# Find midpoints of opposite sides
def find_midpoints_of_opposite_sides(points):
    num_vertices = len(points)
    midpoints = []
    for i in range(num_vertices // 2):
        midpoint1 = (points[i] + points[(i + 1) % num_vertices]) / 2
        midpoint2 = (points[i + num_vertices // 2] + points[(i + num_vertices // 2 + 1) % num_vertices]) / 2
        midpoints.append((midpoint1, midpoint2))
    return midpoints

# Plot shapes only
def plot_shapes_only(ax, shapes, xlim, ylim):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, shape_set in enumerate(shapes):
        c = colours[i % len(colours)]
        for shape in shape_set:
            if shape[0] == 'circle':
                _, center, radius = shape
                circle = plt.Circle(center, radius, color=c, fill=False, linewidth=2)
                ax.add_artist(circle)
            elif shape[0] == 'polygon_with_equal_sides':
                _, points = shape
                ax.plot(points[:, 0], points[:, 1], c=c, linewidth=2)
            else:
                _, points = shape
                ax.plot(points[:, 0], points[:, 1], c=c, linewidth=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plt.title('Corrected Shapes')
    plt.show()

# Plot shapes with symmetry lines
def plot_shapes_and_symmetry(ax, shapes, xlim, ylim):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, shape_set in enumerate(shapes):
        c = colours[i % len(colours)]
        for shape in shape_set:
            if shape[0] == 'circle':
                _, center, radius = shape
                circle = plt.Circle(center, radius, color=c, fill=False, linewidth=2)
                ax.add_artist(circle)
                symmetry_lines = find_symmetry_lines_for_circle(center, radius)
                plot_symmetry_lines(ax, symmetry_lines)
            elif shape[0] == 'polygon_with_equal_sides':
                _, points = shape
                ax.plot(points[:, 0], points[:, 1], c=c, linewidth=2)
                symmetry_lines = find_symmetry_lines_for_polygon(points)
                plot_symmetry_lines(ax, symmetry_lines)
            else:
                _, points = shape
                ax.plot(points[:, 0], points[:, 1], c=c, linewidth=2)
                symmetry_lines = find_symmetry_lines_for_polygon(points)
                plot_symmetry_lines(ax, symmetry_lines)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plt.title('Corrected Shapes with Symmetry Lines')
    plt.show()

# Plot initial shapes
def plot_initial(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, linestyle='--')
    ax.set_aspect('equal')
    plt.show()
    return ax.get_xlim(), ax.get_ylim()

# Plot symmetry lines
def plot_symmetry_lines(ax, symmetry_lines):
    for line in symmetry_lines:
        ax.plot(line[:, 0], line[:, 1], color='grey', linestyle='--', linewidth=1)

# Correct shape by fitting circle or approximating polygon
def correct_shape(points, scale_factor=0.5):
    if is_near_circle(points):
        center, radius = fit_circle(points)
        return 'circle', center, radius
    else:
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        scale = scale_factor * 1000 / np.ptp(points, axis=0).max()
        points = (points * scale).astype(np.int32)
        contour = points.reshape(-1, 1, 2)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if not np.array_equal(approx[0], approx[-1]):
            approx = np.vstack([approx, [approx[0]]])
        corrected_points = np.array([point[0] for point in approx])
        corrected_points = corrected_points / scale  # Scale back to original
        return 'polygon', corrected_points

# Draw shapes on a canvas
def draw_shapes_on_canvas(shapes, canvas_size=(1000, 1000)):
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    for shape_set in shapes:
        for shape in shape_set:
            if shape[0] == 'circle':
                _, center, radius = shape
                center = tuple((np.array(center) * canvas_size[0]).astype(int))
                radius = int(radius * canvas_size[0])
                cv2.circle(canvas, center, radius, 255, -1)  # Draw filled circle
            else:
                _, points = shape
                points = (points * canvas_size[0]).astype(np.int32)
                cv2.fillPoly(canvas, [points], 255)  # Draw filled polygon
    return canvas

# Detect and fill occlusions
def detect_and_fill_occlusions(canvas, original_shapes, canvas_size=(1000, 1000)):
    filled_shapes = []
    for shape_set in original_shapes:
        filled_shape_set = []
        for shape in shape_set:
            if shape[0] == 'circle':
                _, center, radius = shape
                center_scaled = tuple((np.array(center) * canvas_size[0]).astype(int))
                radius_scaled = int(radius * canvas_size[0])
                outline = np.zeros(canvas_size, dtype=np.uint8)
                cv2.circle(outline, center_scaled, radius_scaled, 255, 2)  # Draw circle outline
                missing_parts = cv2.subtract(outline, canvas)
                kernel = np.ones((7, 7), np.uint8)
                dilated_missing_parts = cv2.dilate(missing_parts, kernel, iterations=1)
                canvas = cv2.add(canvas, dilated_missing_parts)
                filled_shape_set.append(('circle', center, radius))
            else:
                _, points = shape
                points_scaled = (points * canvas_size[0]).astype(np.int32)
                outline = np.zeros(canvas_size, dtype=np.uint8)
                cv2.polylines(outline, [points_scaled], isClosed=True, color=255, thickness=2)
                missing_parts = cv2.subtract(outline, canvas)
                kernel = np.ones((7, 7), np.uint8)
                dilated_missing_parts = cv2.dilate(missing_parts, kernel, iterations=1)
                canvas = cv2.add(canvas, dilated_missing_parts)
                filled_shape_set.append(('polygon', points))
        filled_shapes.append(filled_shape_set)
    return filled_shapes

# Save a shape as PNG
def save_shape_as_png(shape, buffer, canvas_size=(1000, 1000)):
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255  # 3 channels for RGB, white background
    if shape[0] == 'circle':
        _, center, radius = shape
        center = tuple((np.array(center) * canvas_size[0]).astype(int))
        radius = int(radius * canvas_size[0])
        cv2.circle(canvas, center, radius, (0, 0, 0), -1)  # Draw filled circle in black
    else:
        _, points = shape
        points = (points * canvas_size[0]).astype(np.int32)
        cv2.fillPoly(canvas, [points], (0, 0, 0))  # Draw filled polygon in black

    success, encoded_image = cv2.imencode('.png', canvas)
    if success:
        buffer.write(encoded_image.tobytes())

# Save a shape as SVG
def save_shape_as_svg(shape, buffer, canvas_size=(1000, 1000)):
    dwg = svgwrite.Drawing(size=(canvas_size[0], canvas_size[1]))
    if shape[0] == 'circle':
        _, center, radius = shape
        center_x = center[0] * canvas_size[0]
        center_y = center[1] * canvas_size[1]
        radius = radius * canvas_size[0]
        dwg.add(dwg.circle(center=(center_x, center_y), r=radius, fill='black'))
    else:
        _, points = shape
        points = points * canvas_size[0]
        points = [(point[0], point[1]) for point in points]
        dwg.add(dwg.polygon(points=points, fill='black'))

    svg_string = dwg.tostring()
    buffer.write(svg_string.encode('utf-8'))




# Streamlit interface
def streamlit_interface():
    st.title("Shape Processing and Symmetry Detection")
    st.write("Upload a CSV file with shape data to process and optionally display symmetry lines.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    show_symmetry_lines = st.checkbox("Show Symmetry Lines", value=False)

    if uploaded_file:
        csv_path = uploaded_file.read()  # Read the file as bytes
        paths_XYs = read_csv(BytesIO(csv_path))
        xlim, ylim = plot_initial(paths_XYs)

        corrected_shapes = []
        for shape in paths_XYs:
            corrected_shape = []
            for points in shape:
                corrected_shape.append(correct_shape(np.array(points), scale_factor=0.5))
            corrected_shapes.append(corrected_shape)

        canvas = draw_shapes_on_canvas(corrected_shapes)
        final_corrected_shapes = detect_and_fill_occlusions(canvas, corrected_shapes)

        buf = BytesIO()
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
        if show_symmetry_lines:
            plot_shapes_and_symmetry(ax, final_corrected_shapes, xlim, ylim)
        else:
            plot_shapes_only(ax, final_corrected_shapes, xlim, ylim)
        plt.savefig(buf, format='png')
        buf.seek(0)

        st.image(buf, caption="Processed Shapes", use_column_width=True)

        st.write("Download each shape as PNG or SVG:")
        for i, shape_set in enumerate(final_corrected_shapes):
            for j, shape in enumerate(shape_set):
                shape_name = f"shape_{i}_{j}"
                png_buffer = BytesIO()
                svg_buffer = BytesIO()
                save_shape_as_png(shape, png_buffer)
                save_shape_as_svg(shape, svg_buffer)

                st.download_button(f"Download {shape_name}.png", data=png_buffer, file_name=f"{shape_name}.png")
                st.download_button(f"Download {shape_name}.svg", data=svg_buffer, file_name=f"{shape_name}.svg")

if __name__ == "__main__":
    streamlit_interface()
