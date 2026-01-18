import numpy as np
PIXEL_SIZE = 0.00267857 # The size of each pixel in real-world units (such as meters)

BOUNDS = np.float32([[-0.2, 0.2], [-0.7, -0.3], [0, 0.15]])  # (X, Y, Z) (width, depth, height)

JOINT_LIMITS = [[- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi], [-np.pi, np.pi],
				[- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi], [- 2 * np.pi, 2 * np.pi]]

COLORS = {
	'black': (0, 0, 0),
	'white': (255, 255, 255),
	'blue': (78 / 255, 121 / 255, 167 / 255, 255 / 255),
	'red': (255 / 255, 87 / 255, 89 / 255, 255 / 255),
	'green': (89 / 255, 169 / 255, 79 / 255, 255 / 255),
	'orange': (242 / 255, 142 / 255, 43 / 255, 255 / 255),
	'yellow': (237 / 255, 201 / 255, 72 / 255, 255 / 255),
	'purple': (176 / 255, 122 / 255, 161 / 255, 255 / 255),
	'pink': (255 / 255, 157 / 255, 167 / 255, 255 / 255),
	'cyan': (118 / 255, 183 / 255, 178 / 255, 255 / 255),
	'brown': (156 / 255, 117 / 255, 95 / 255, 255 / 255),
	'gray': (186 / 255, 176 / 255, 172 / 255, 255 / 255),
}

PICK_TARGETS = {
	'red apple': None,
	'red strawberry': None,
	'yellow banana': None,
	'pink peach': None,
	'white egg': None,
	'green pear': None,
	'orange orange': None,
	'yellow lemon': None,
	'black spoon': None,
	'black spatula': None,
	'black knife': None,
	'red gelatin_box': None,

	'blue block': None,
	'red block': None,
	'green block': None,
	'orange block': None,
	'yellow block': None,
	'purple block': None,
	'pink block': None,
	'cyan block': None,
	'brown block': None,
	'gray block': None,
}

PLACE_TARGETS = {
	'blue block': None,
	'red block': None,
	'red gelatin_box': None,
	'green block': None,
	'orange block': None,
	'yellow block': None,
	'purple block': None,
	'pink block': None,
	'cyan block': None,
	'brown block': None,
	'gray block': None,

	'blue bowl': None,
	'red bowl': None,
	'green bowl': None,
	'orange bowl': None,
	'yellow bowl': None,
	'purple bowl': None,
	'pink bowl': None,
	'cyan bowl': None,
	'brown bowl': None,
	'gray bowl': None,

	'top left corner': (-0.2, -0.3, 0),
	'top side': (0, -0.3, 0),
	'top right corner': (0.2, -0.3, 0),
	'left side': (-0.2, -0.5, 0),
	'middle': (0, -0.5, 0),
	'right side': (0.2, -0.5, 0),
	'bottom left corner': (-0.2, -0.7, 0),
	'bottom side': (0, -0.7, 0),
	'bottom right corner': (0.2, -0.7, 0),
}

FIXED_DESTINATION = {
	'top left corner': (-0.2, -0.3, 0),
	'top side': (0, -0.3, 0),
	'top right corner': (0.2, -0.3, 0),
	'left side': (-0.2, -0.5, 0),
	'middle': (0, -0.5, 0),
	'right side': (0.2, -0.5, 0),
	'bottom left corner': (-0.2, -0.7, 0),
	'bottom side': (0, -0.7, 0),
	'bottom right corner': (0.2, -0.7, 0),
}

INSTRUCTION_FORM = [
	"Touch the {}",
	"Pick the {} up",
	"Move the {} to the {}",
	"Put the {} onto the {}",
]

ACTION_STATE = {
	'Touch': '0',
	'Pick': '1',
	'Move': '2',
	'Put': '3',
}
