
#include <vector>
#include <map>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <set>

#include "common.h"

#define NULL nullptr

using namespace std;

// a trie data structure for fast matching subsequences

class FNode
{
private:
    
	void initialize_fnode()
	{		
		feature_index = -1;
        
	}

public:
	int feature_index;
	map<char, FNode *> children;

	

	FNode()
	{
		initialize_fnode();
	}

	~FNode()
	{
		for (map<char, FNode *>::iterator itr = children.begin(); itr != children.end(); itr++)
		{
			delete itr->second;
		}		
	}

	

	FNode *get_child(char c)
	{
		if (children[c] == NULL)
		{
			children[c] = new FNode();
			// children[c]->ngram = ngram + c;
		}
		return children[c];
	}

	FNode *get_child_without_creating(char c)
	{
		return children[c];
	} 
};

class SeqTrie
{
private:
    FNode * root;
    int num_of_features;
    void build(vector<string> sequences)
    {
        root = new FNode();
        for (int i = 0; i < sequences.size(); i++){
            string s = sequences[i];
            FNode* current_node = root;
            for (char c: s){
                current_node = current_node->get_child(c);
            }
            current_node->feature_index = i;
        }
    }

    

public:
    SeqTrie(vector<string> sequences){
        num_of_features = sequences.size();
        build(sequences);
    }

    

    vector<int> search(string sequence){
        vector<int> count(num_of_features,0);

        vector<FNode*> current_nodes;
        current_nodes.push_back(root);

        for (char c: sequence){
            vector<FNode*> next_nodes;
            next_nodes.push_back(root);

            for (FNode* n: current_nodes){
                FNode* next = n->get_child_without_creating(c);
                if (next != NULL){
                    if (next->feature_index >= 0){
                        count[next->feature_index]++;
                    }
                    next_nodes.push_back(next);
                }
            }

            current_nodes = next_nodes;



        }

        return count;


    }



};