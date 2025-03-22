To interface with the game's memory, we utilized the implementation of [SuperHexagonAI](https://github.com/polarbart/SuperHexagonAI).
This project provides a Python wrapper for the C++ memory hook developed in [super-hexagon-ai](https://github.com/adrianchifor/super-hexagon-ai).
With this wrapper, we accessed the game's memory, enabling features like:

- Starting the game
- Speeding up the game
- Freezing the game
- Stepping one frame at a time
- Selecting a level
- Restarting a level

With the help of the wrapper and provided functions, it is also possible to carry out memory manipulations on the game:

- ::: superhexagon.SuperHexagonInterface._left
- ::: superhexagon.SuperHexagonInterface._right
- ::: superhexagon.SuperHexagonInterface.get_triangle_angle

These functions provided a solid foundation but had limitations requiring deeper insights into the game's state. To overcome them, we extended the wrapper by identifying additional memory locations from the C++ implementation, adding the following methods:

- ::: superhexagon.SuperHexagonInterface.get_num_slots
- ::: superhexagon.SuperHexagonInterface.get_num_walls
- ::: superhexagon.SuperHexagonInterface.get_triangle_slot
- ::: superhexagon.SuperHexagonInterface.get_walls

These enhancements enabled us to gather all necessary data for observations and reward calculation.