from typing import Set, List

from classes.dataset import Dataset
from classes.itemset import Itemset
from classes.item import Item
from classes.itemsets_with_occurrence_counts import ItemsetsWithOccurrenceCounts
from classes.sorted_dataset import SortedDataset
from classes.sorted_transaction import SortedTransaction
from classes.item_tuple import ItemTuple
from classes.fp_tree import FPTree
from classes.conditional_pattern_base import ConditionalPatternBase
from classes.conditional_pattern import ConditionalPattern


class FPgrowth:
    def __init__(self, min_support: int = 2):
        """
        Initialize the FP-growth algorithm with a minimum (absolute) support.
        """
        if not isinstance(min_support, int) or min_support < 1:
            raise ValueError("The minimum support must be a positive integer.")

        self.min_support = min_support
        self.frequent_itemsets = set()

    def _generate_frequent_one_itemsets_with_occurrence_counts(
        self, dataset: Dataset
    ) -> ItemsetsWithOccurrenceCounts:
        """Generate all frequent 1-itemsets for the given dataset."""
        item_counts = {}
        
        for transaction in dataset.transactions:
            for item in transaction.items:
                itemset = Itemset(frozenset({item}))
                item_counts[itemset] = item_counts.get(itemset, 0) + 1
        
        result = ItemsetsWithOccurrenceCounts({})
        for itemset, count in item_counts.items():
            if count >= self.min_support:
                result[itemset] = count
        
        return result

    def _generate_f_list(
        self, frequent_one_itemsets: ItemsetsWithOccurrenceCounts
    ) -> List[Itemset]:
        """Generate the f-list sorted by decreasing occurrence count."""
        sorted_items = sorted(
            frequent_one_itemsets.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [itemset for itemset, _ in sorted_items]

    def _sort_dataset_according_to_f_list(
        self, dataset: Dataset, f_list: List[Itemset]
    ) -> SortedDataset:
        """Sort the dataset according to the given f-list."""
        item_order = {}
        for idx, itemset in enumerate(f_list):
            item = next(iter(itemset.items))
            item_order[item] = idx
        
        sorted_transactions = set()
        
        for transaction in dataset.transactions:
            filtered_items = [
                item for item in transaction.items 
                if item in item_order
            ]
            filtered_items.sort(key=lambda x: item_order[x])
            
            item_tuple = ItemTuple(tuple(filtered_items))
            sorted_transaction = SortedTransaction(transaction.id, item_tuple)
            sorted_transactions.add(sorted_transaction)
        
        return SortedDataset(frozenset(sorted_transactions))

    def _construct_initial_fp_tree(self, sorted_dataset: SortedDataset) -> FPTree:
        """Construct the initial FP-tree from the given sorted dataset."""
        fp_tree = FPTree()
        
        for transaction in sorted_dataset.transactions:
            if transaction.items.items:
                fp_tree.add_items_to_tree(transaction.items, 1)
        
        return fp_tree

    def _get_conditional_pattern_base(
        self, item: Item, fp_tree: FPTree
    ) -> ConditionalPatternBase:
        """Get the conditional pattern base for the given item in the FP-tree."""
        patterns = []
        
        header_table = fp_tree.get_header_table()
        
        # Find header element for this item
        target_element = None
        for element in header_table.elements:
            if element.item == item:
                target_element = element
                break
        
        if target_element is None:
            return ConditionalPatternBase(patterns)
        
        # Traverse all node links for this item
        for node in target_element.node_links:
            predecessors = node.get_predecessors()
            
            if predecessors:
                # Extract Item objects from the predecessor nodes
                items_list = []
                for pred_node in predecessors:
                    items_list.append(pred_node.item)
                
                prefix_tuple = ItemTuple(tuple(items_list))
                pattern = ConditionalPattern(prefix_tuple, node.occurrence_count)
                patterns.append(pattern)
        
        return ConditionalPatternBase(patterns)

    def _construct_conditional_fp_tree(
        self, conditional_pattern_base: ConditionalPatternBase
    ) -> FPTree:
        """
        Construct a conditional FP-tree from the given conditional pattern base.
        This method adds all patterns without filtering (for test_construct_conditional_fp_tree).
        """
        conditional_tree = FPTree()
        
        for pattern in conditional_pattern_base.conditional_patterns:
            if pattern.prefix_items.items:
                conditional_tree.add_items_to_tree(
                    pattern.prefix_items,
                    pattern.occurrence_count
                )
        
        return conditional_tree

    def _filter_conditional_pattern_base(
        self, conditional_pattern_base: ConditionalPatternBase
    ) -> ConditionalPatternBase:
        """
        Filter the conditional pattern base to only include items that meet min_support.
        This is used during recursive mining.
        """
        # Count occurrences of each item in the conditional patterns
        item_counts = {}
        
        for pattern in conditional_pattern_base.conditional_patterns:
            for item in pattern.prefix_items.items:
                item_counts[item] = item_counts.get(item, 0) + pattern.occurrence_count
        
        # Determine which items are frequent
        frequent_items = {item for item, count in item_counts.items() if count >= self.min_support}
        
        # Create filtered patterns
        filtered_patterns = []
        for pattern in conditional_pattern_base.conditional_patterns:
            filtered_items = [item for item in pattern.prefix_items.items if item in frequent_items]
            if filtered_items:
                filtered_pattern = ConditionalPattern(
                    ItemTuple(tuple(filtered_items)),
                    pattern.occurrence_count
                )
                filtered_patterns.append(filtered_pattern)
        
        return ConditionalPatternBase(filtered_patterns)

    def _fp_growth(self, fp_tree: FPTree, prefix: Itemset):
        """
        Recursive FP-growth mining algorithm.
        """
        header_table = fp_tree.get_header_table()
        
        # Process each item in the header table from bottom to top
        for element in reversed(header_table.elements):
            item = element.item
            
            # Create new frequent itemset by adding current item to prefix
            new_items = set(prefix.items) | {item}
            new_itemset = Itemset(frozenset(new_items))
            
            # Add to global frequent itemsets
            self.frequent_itemsets.add(new_itemset)
            
            # Build conditional pattern base for this item
            conditional_base = self._get_conditional_pattern_base(item, fp_tree)
            
            # Filter the conditional pattern base to only include items that meet min_support
            filtered_base = self._filter_conditional_pattern_base(conditional_base)
            
            # Build conditional FP-tree from the filtered pattern base
            conditional_tree = self._construct_conditional_fp_tree(filtered_base)
            
            # Recursively mine if the conditional tree is not empty
            if not conditional_tree.is_empty():
                self._fp_growth(conditional_tree, new_itemset)

    def fit(self, dataset: Dataset):
        """
        Use the FP-growth algorithm to find all frequent itemsets.
        """
        # Reset frequent itemsets
        self.frequent_itemsets = set()
        
        # Step 1: Generate frequent 1-itemsets with their occurrence counts
        frequent_one_itemsets = self._generate_frequent_one_itemsets_with_occurrence_counts(dataset)
        
        # If no frequent itemsets, return
        if not frequent_one_itemsets:
            return
        
        # Step 2: Generate the f-list (sorted by descending support)
        f_list = self._generate_f_list(frequent_one_itemsets)
        
        # Step 3: Sort the dataset according to the f-list
        sorted_dataset = self._sort_dataset_according_to_f_list(dataset, f_list)
        
        # Step 4: Build the initial FP-tree
        fp_tree = self._construct_initial_fp_tree(sorted_dataset)
        
        # Step 5: Add all frequent 1-itemsets to the result
        for itemset in frequent_one_itemsets.keys():
            self.frequent_itemsets.add(itemset)
        
        # Step 6: Recursively mine the FP-tree to find larger itemsets
        if not fp_tree.is_empty():
            self._fp_growth(fp_tree, Itemset(frozenset()))