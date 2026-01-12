//! Goal hierarchy tree structure.

use super::error::GoalHierarchyError;
use super::level::GoalLevel;
use super::node::GoalNode;
use std::collections::HashMap;
use uuid::Uuid;

/// Goal hierarchy tree structure.
///
/// Manages a tree of goals with a single North Star at the root.
/// Used for hierarchical alignment propagation in purpose vector computation.
///
/// # Invariants
///
/// - At most one North Star goal
/// - All child goals have valid parent references
/// - No cycles in the hierarchy
#[derive(Clone, Debug, Default)]
pub struct GoalHierarchy {
    pub(crate) nodes: HashMap<Uuid, GoalNode>,
    north_star: Option<Uuid>,
}

impl GoalHierarchy {
    /// Create a new empty goal hierarchy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a goal to the hierarchy.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Adding a second North Star goal
    /// - Child goal's parent doesn't exist
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_core::purpose::{GoalHierarchy, GoalNode, GoalLevel, GoalDiscoveryMetadata};
    /// use context_graph_core::types::fingerprint::SemanticFingerprint;
    ///
    /// let mut hierarchy = GoalHierarchy::new();
    ///
    /// let discovery = GoalDiscoveryMetadata::bootstrap();
    /// let fp = SemanticFingerprint::zeroed();
    /// let ns = GoalNode::autonomous_goal(
    ///     "North Star".into(),
    ///     GoalLevel::NorthStar,
    ///     fp,
    ///     discovery,
    /// ).unwrap();
    /// hierarchy.add_goal(ns).unwrap();
    /// ```
    pub fn add_goal(&mut self, goal: GoalNode) -> Result<(), GoalHierarchyError> {
        // Validate parent exists (except for NorthStar)
        if let Some(ref parent_id) = goal.parent_id {
            if !self.nodes.contains_key(parent_id) {
                return Err(GoalHierarchyError::ParentNotFound(*parent_id));
            }
        }

        // Only one North Star allowed
        if goal.level == GoalLevel::NorthStar {
            if self.north_star.is_some() {
                return Err(GoalHierarchyError::MultipleNorthStars);
            }
            self.north_star = Some(goal.id);
        }

        // Update parent's child list
        if let Some(parent_id) = goal.parent_id {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.add_child(goal.id);
            }
        }

        self.nodes.insert(goal.id, goal);
        Ok(())
    }

    /// Get the North Star goal.
    ///
    /// Returns None if no North Star has been added.
    pub fn north_star(&self) -> Option<&GoalNode> {
        self.north_star.and_then(|id| self.nodes.get(&id))
    }

    /// Check if a North Star goal exists.
    #[inline]
    pub fn has_north_star(&self) -> bool {
        self.north_star.is_some()
    }

    /// Get a goal by ID.
    pub fn get(&self, id: &Uuid) -> Option<&GoalNode> {
        self.nodes.get(id)
    }

    /// Get direct children of a goal.
    pub fn children(&self, parent_id: &Uuid) -> Vec<&GoalNode> {
        self.get(parent_id)
            .map(|parent| {
                parent
                    .child_ids
                    .iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all goals at a specific level.
    pub fn at_level(&self, level: GoalLevel) -> Vec<&GoalNode> {
        self.nodes.values().filter(|n| n.level == level).collect()
    }

    /// Total number of goals in the hierarchy.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the hierarchy is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all goals.
    pub fn iter(&self) -> impl Iterator<Item = &GoalNode> {
        self.nodes.values()
    }

    /// Get path from a goal to the North Star.
    ///
    /// Returns the sequence of goal IDs from the given goal up to (and including)
    /// the North Star. Returns empty vec if goal not found.
    pub fn path_to_north_star(&self, goal_id: &Uuid) -> Vec<Uuid> {
        let mut path = Vec::new();
        let mut current = self.nodes.get(goal_id);

        while let Some(node) = current {
            path.push(node.id);
            current = node.parent_id.and_then(|pid| self.nodes.get(&pid));
        }

        path
    }

    /// Validate hierarchy integrity.
    ///
    /// Checks:
    /// - North Star exists if hierarchy is not empty
    /// - All parent references are valid
    pub fn validate(&self) -> Result<(), GoalHierarchyError> {
        if self.north_star.is_none() && !self.nodes.is_empty() {
            return Err(GoalHierarchyError::NoNorthStar);
        }

        // Check all parents exist
        for node in self.nodes.values() {
            if let Some(ref parent_id) = node.parent_id {
                if !self.nodes.contains_key(parent_id) {
                    return Err(GoalHierarchyError::ParentNotFound(*parent_id));
                }
            }
        }

        Ok(())
    }
}
