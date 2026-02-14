"""
Generate a professional table figure showing comparison of image data
before and after augmentation for the Mulberry Disease Prediction project.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def generate_augmentation_table():
    """Generate Table 3: Comparison of Image Data Before and After Image Augmentation."""

    # Disease classes and data counts
    classes = [
        "Healthy Leaves",
        "Rust Leaves",
        "Spot Leaves",
        "Deformed Leaves",
        "Yellow Leaves",
    ]

    before_count = 200  # images per class before augmentation
    after_count = 1200   # images per class after augmentation (6x)
    num_classes = len(classes)
    total_before = before_count * num_classes
    total_after = after_count * num_classes

    # Table data
    col_labels = ["Class", "Before\nAugmentation", "After\nAugmentation"]

    cell_data = []
    for cls in classes:
        cell_data.append([cls, f"{before_count:,} images", f"{after_count:,} images"])
    cell_data.append(["Total", f"{total_before:,} images", f"{total_after:,} images"])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis('off')

    # Title
    fig.text(
        0.5, 0.93,
        "Table 3. Comparison of Image Data Before and After Image Augmentation",
        ha='center', va='center',
        fontsize=13, fontweight='bold',
        fontfamily='serif'
    )

    # Create the table
    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.36, 0.28, 0.28]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Header styling
    header_color = '#2C3E50'
    header_text_color = 'white'
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight='bold', fontfamily='serif')
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)

    # Data row styling
    row_colors = ['#F8F9FA', '#FFFFFF']
    for i in range(1, len(cell_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_edgecolor('#DEE2E6')
            cell.set_linewidth(1.0)
            cell.set_text_props(fontfamily='serif')

            if i == len(cell_data):  # Total row
                cell.set_facecolor('#D5E8D4')
                cell.set_text_props(fontweight='bold', fontfamily='serif')
            else:
                cell.set_facecolor(row_colors[(i - 1) % 2])

    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.08, right=0.92)

    # Save
    output_path = "augmentation_comparison_table.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Table saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_augmentation_table()
