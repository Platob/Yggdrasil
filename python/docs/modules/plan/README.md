# Execution Plans

SQL parsing, lazy execution, and Arrow-native UDF execution.

## API Reference

::: yggdrasil.plan
    options:
      show_root_heading: false
      members:
        - parse_sql
        - ExecutionPlan
        - SelectPlan
        - LazyTabular
        - PlanNode
        - SelectNode
        - InsertNode
        - MergeNode
        - ScanNode
        - FunctionRegistry
        - FunctionMeta
        - BUILTIN_REGISTRY

## Function Registry

::: yggdrasil.plan.func_registry
    options:
      show_root_heading: false
      members:
        - FunctionRegistry
        - FunctionMeta
        - BUILTIN_REGISTRY
        - explode_table
        - posexplode_table

## SQL Parser

::: yggdrasil.plan.sql_parser
    options:
      show_root_heading: false
      members:
        - parse_sql
        - SQLQueryParser

## Plan Nodes

::: yggdrasil.plan.nodes
    options:
      show_root_heading: false
      members:
        - PlanNode
        - SelectNode
        - InsertNode
        - MergeNode
        - ScanNode

## Operations

::: yggdrasil.plan.ops
    options:
      show_root_heading: false
