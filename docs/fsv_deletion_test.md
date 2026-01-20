# FSV Deletion Test (Tracked File)

**Unique Marker:** UNIQUE_MARKER_DELETION_TEST_DOLPHIN_99999

## Purpose

This file will be committed to git, then deleted to test proper cleanup.

The file watcher should detect the deletion via `git status --porcelain` showing ` D` status.
