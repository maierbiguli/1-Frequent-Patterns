from typing import Set

from classes.dataset import Dataset
from classes.itemset import Itemset
from classes.itemsets_with_occurrence_counts import ItemsetsWithOccurrenceCounts


class Apriori:
    def __init__(self, min_support: int = 2):
        """
        Initialize the Apriori algorithm with the a minimum (absolute) support.

        Parameters:
        min_support (int): The minimum (absolute) support. This parameter defines the minimum number
                           of occurrences an itemset must have to be considered frequent. Must be a positive integer.
                           Default value is 2.
        """
        # Ensure that the minimum support is a positive integer
        if not isinstance(min_support, int) or min_support < 1:
            raise ValueError("The minimum support must be a positive integer.")

        self.min_support = min_support
        self.frequent_itemsets = set()

    def _generate_one_itemsets(self, dataset: Dataset) -> Set[Itemset]:
        """
        Generate all 1-itemsets for the given dataset.

        Parameters:
        dataset (Dataset): The dataset for which the 1-itemsets should be generated.

        Returns:
        Set[Itemset]: A set containing all 1-itemsets that are contained in the dataset.
        """
        one_itemsets = set()
        
        for transaction in dataset.transactions:
            for item in transaction.items:
                itemset = Itemset(frozenset({item}))
                one_itemsets.add(itemset)
        
        return one_itemsets

    def _count_occurrences_of_itemsets(
        self, dataset: Dataset, itemsets: Set[Itemset]
    ) -> ItemsetsWithOccurrenceCounts:
        """
        Count the occurrences of the given itemsets in the dataset.

        Parameters:
        dataset (Dataset): The dataset for which the itemset occurrences should be counted.
        itemsets (Set[Itemset]): The itemsets for which the occurrences should be counted. The itemsets do not need to be present in the dataset.

        Returns:
        ItemsetsWithOccurrenceCounts: A dictionary containing the itemsets as keys and their occurrence counts as values.
        """
        # Initialize the result dictionary with an empty dict
        counts = ItemsetsWithOccurrenceCounts({})
        
        # Initialize all itemsets with count 0
        for itemset in itemsets:
            counts[itemset] = 0
        
        # Count occurrences
        for transaction in dataset.transactions:
            for itemset in itemsets:
                if itemset.items.issubset(transaction.items.items):
                    counts[itemset] += 1
        
        return counts

    def _prune_itemsets_below_min_support(
        self,
        itemsets_with_occurrence_counts: ItemsetsWithOccurrenceCounts,
    ) -> Set[Itemset]:
        """
        Prune itemsets that are below the minimum support threshold.

        Parameters:
        itemsets_with_occurrence_counts (ItemsetsWithOccurrenceCounts): A dictionary containing the itemsets as keys and their occurrence counts as values.

        Returns:
        Set[Itemset]: A set containing all itemsets that are considered frequent.
        """
        frequent_itemsets = set()
        
        for itemset, count in itemsets_with_occurrence_counts.items():
            if count >= self.min_support:
                frequent_itemsets.add(itemset)
        
        return frequent_itemsets

    def _generate_candidate_itemsets(
        self, frequent_itemsets: Set[Itemset]
    ) -> Set[Itemset]:
        """
        Generate length-k+1 candidate itemsets based on the given frequent itemsets.
        k is the length of the longest frequent itemset.

        Parameters:
        frequent_itemsets (Set[Itemset]): A set containing all frequent itemsets.

        Returns:
        Set[Itemset]: A set containing all length-k+1 candidate itemsets.
        """
        # If there are no frequent itemsets, return an empty set
        if not frequent_itemsets:
            return set()
        
        # Find the maximum length of itemsets
        # In Apriori, we should only use itemsets of the same length to generate candidates
        max_length = max(len(itemset.items) for itemset in frequent_itemsets)
        
        # Filter to only itemsets of max_length
        # This is key: we only use the longest frequent itemsets to generate the next candidates
        filtered_itemsets = {itemset for itemset in frequent_itemsets if len(itemset.items) == max_length}
        
        # If we have less than 2 itemsets of this length, cannot generate candidates
        if len(filtered_itemsets) < 2:
            return set()
        
        k = max_length
        candidates = set()
        itemset_list = list(filtered_itemsets)
        
        for i in range(len(itemset_list)):
            for j in range(i + 1, len(itemset_list)):
                itemset1 = itemset_list[i]
                itemset2 = itemset_list[j]
                
                # Get sorted lists of items for comparison
                items1 = sorted(list(itemset1.items), key=lambda x: x.name)
                items2 = sorted(list(itemset2.items), key=lambda x: x.name)
                
                # Check if the first k-1 items are the same
                if items1[:k-1] == items2[:k-1]:
                    # Merge the two itemsets
                    new_items = set(itemset1.items) | set(itemset2.items)
                    candidate = Itemset(frozenset(new_items))
                    
                    # Apriori pruning: Check if all (k)-subsets are frequent
                    # For generating (k+1)-itemset, we need to check all k-subsets
                    is_valid = True
                    for item_to_remove in candidate.items:
                        subset_items = set(candidate.items) - {item_to_remove}
                        subset = Itemset(frozenset(subset_items))
                        if subset not in frequent_itemsets:
                            is_valid = False
                            break
                    
                    if is_valid:
                        candidates.add(candidate)
        
        return candidates

    def fit(self, dataset: Dataset):
        """
        Use the Apriori algorithm to find all frequent itemsets in the given dataset.
        Saves the frequent itemsets in the frequent_itemsets attribute.

        Parameters:
        dataset (Dataset): The dataset to which the Apriori algorithm should be fitted.
        """
        # Reset the set of frequent itemsets
        self.frequent_itemsets = set()
        
        # Step 1: Generate 1-itemsets
        current_itemsets = self._generate_one_itemsets(dataset)
        
        # Step 2: Iteratively generate larger itemsets until no more candidates are found
        while current_itemsets:
            # Count occurrences of current itemsets
            counts = self._count_occurrences_of_itemsets(dataset, current_itemsets)
            
            # Prune itemsets below minimum support
            frequent = self._prune_itemsets_below_min_support(counts)
            
            # Add frequent itemsets to the result
            self.frequent_itemsets.update(frequent)
            
            # Generate candidate itemsets for the next iteration
            current_itemsets = self._generate_candidate_itemsets(frequent)