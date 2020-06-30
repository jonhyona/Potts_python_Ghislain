from suffix_trees import STree

# Suffix-Tree example.
st = STree.STree("abcdefghab")
print(st.find("abc")) # 0
print(st.find_all("ab")) # {0, 8}

# Generalized Suffix-Tree example.
a = ["xxxabcxxx", "adsaabc", "ytysabcrew", "qqqabcqw", "aaabc"]
st = STree.STree(a)
print(st.lcs()) # "abc"
