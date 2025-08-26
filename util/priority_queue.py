class PriorityQueue:

    def __init__(self):
        self._items = []
        self._priorities = []

    def put(self, item, priority: float) -> None:
        """
        Put an element into the queue.

        :param item: An object of any type.
        :param priority: The priority of `item` represented as a float, where a smaller number has a higher priority.
        """
        self._items.append(item)
        self._priorities.append(priority)
        element_index = len(self._items) - 1
        if element_index == 0:
            # Only element in the priority queue.
            return

        while self._priorities[element_index] < self._priorities[element_index - 1]:
            # Move element towards front of list
            self._items[element_index - 1], self._items[element_index] = self._items[element_index], self._items[element_index - 1]
            self._priorities[element_index - 1], self._priorities[element_index] = self._priorities[element_index], self._priorities[element_index - 1]
            if element_index - 1 == 0:
                return
            
    def get(self) -> object:
        """
        Retrieve the next item in the priority queue.
        
        :return: The item in the priority queue with the lowest priority score.
        """
        item = self._items[0]
        self._items = self._items[1:]
        self._priorities = self._priorities[1:]
        return item

    def empty(self) -> bool:
        """
        Checks if the priority queue is empty.

        :return: True if the priority queue is empty and False otherwise.
        """
        if self._items == []:
            return True
        else:
            return False
    
