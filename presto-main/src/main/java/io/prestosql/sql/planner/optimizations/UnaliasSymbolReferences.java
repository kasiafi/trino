/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.prestosql.sql.planner.optimizations;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import io.prestosql.Session;
import io.prestosql.execution.warnings.WarningCollector;
import io.prestosql.metadata.Metadata;
import io.prestosql.spi.connector.ColumnHandle;
import io.prestosql.sql.planner.DeterminismEvaluator;
import io.prestosql.sql.planner.OrderingScheme;
import io.prestosql.sql.planner.PartitioningScheme;
import io.prestosql.sql.planner.PlanNodeIdAllocator;
import io.prestosql.sql.planner.Symbol;
import io.prestosql.sql.planner.SymbolAllocator;
import io.prestosql.sql.planner.TypeProvider;
import io.prestosql.sql.planner.plan.AggregationNode;
import io.prestosql.sql.planner.plan.ApplyNode;
import io.prestosql.sql.planner.plan.AssignUniqueId;
import io.prestosql.sql.planner.plan.Assignments;
import io.prestosql.sql.planner.plan.CorrelatedJoinNode;
import io.prestosql.sql.planner.plan.DeleteNode;
import io.prestosql.sql.planner.plan.DistinctLimitNode;
import io.prestosql.sql.planner.plan.EnforceSingleRowNode;
import io.prestosql.sql.planner.plan.ExceptNode;
import io.prestosql.sql.planner.plan.ExchangeNode;
import io.prestosql.sql.planner.plan.ExplainAnalyzeNode;
import io.prestosql.sql.planner.plan.FilterNode;
import io.prestosql.sql.planner.plan.GroupIdNode;
import io.prestosql.sql.planner.plan.IndexJoinNode;
import io.prestosql.sql.planner.plan.IndexSourceNode;
import io.prestosql.sql.planner.plan.IntersectNode;
import io.prestosql.sql.planner.plan.JoinNode;
import io.prestosql.sql.planner.plan.LimitNode;
import io.prestosql.sql.planner.plan.MarkDistinctNode;
import io.prestosql.sql.planner.plan.OffsetNode;
import io.prestosql.sql.planner.plan.OutputNode;
import io.prestosql.sql.planner.plan.PlanNode;
import io.prestosql.sql.planner.plan.PlanVisitor;
import io.prestosql.sql.planner.plan.ProjectNode;
import io.prestosql.sql.planner.plan.RemoteSourceNode;
import io.prestosql.sql.planner.plan.RowNumberNode;
import io.prestosql.sql.planner.plan.SampleNode;
import io.prestosql.sql.planner.plan.SemiJoinNode;
import io.prestosql.sql.planner.plan.SortNode;
import io.prestosql.sql.planner.plan.SpatialJoinNode;
import io.prestosql.sql.planner.plan.StatisticsWriterNode;
import io.prestosql.sql.planner.plan.TableDeleteNode;
import io.prestosql.sql.planner.plan.TableFinishNode;
import io.prestosql.sql.planner.plan.TableScanNode;
import io.prestosql.sql.planner.plan.TableWriterNode;
import io.prestosql.sql.planner.plan.TopNNode;
import io.prestosql.sql.planner.plan.TopNRowNumberNode;
import io.prestosql.sql.planner.plan.UnionNode;
import io.prestosql.sql.planner.plan.UnnestNode;
import io.prestosql.sql.planner.plan.ValuesNode;
import io.prestosql.sql.planner.plan.WindowNode;
import io.prestosql.sql.tree.Expression;
import io.prestosql.sql.tree.NullLiteral;
import io.prestosql.sql.tree.SymbolReference;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static io.prestosql.sql.planner.plan.JoinNode.Type.INNER;
import static java.util.Objects.requireNonNull;

/**
 * Re-maps symbol references that are just aliases of each other (e.g., due to projections like {@code $0 := $1})
 * <p/>
 * E.g.,
 * <p/>
 * {@code Output[$0, $1] -> Project[$0 := $2, $1 := $3 * 100] -> Aggregate[$2, $3 := sum($4)] -> ...}
 * <p/>
 * gets rewritten as
 * <p/>
 * {@code Output[$2, $1] -> Project[$2, $1 := $3 * 100] -> Aggregate[$2, $3 := sum($4)] -> ...}
 */
public class UnaliasSymbolReferences
        implements PlanOptimizer
{
    private final Metadata metadata;

    public UnaliasSymbolReferences(Metadata metadata)
    {
        this.metadata = requireNonNull(metadata, "metadata is null");
    }

    @Override
    public PlanNode optimize(PlanNode plan, Session session, TypeProvider types, SymbolAllocator symbolAllocator, PlanNodeIdAllocator idAllocator, WarningCollector warningCollector)
    {
        requireNonNull(plan, "plan is null");
        requireNonNull(session, "session is null");
        requireNonNull(types, "types is null");
        requireNonNull(symbolAllocator, "symbolAllocator is null");
        requireNonNull(idAllocator, "idAllocator is null");

        return plan.accept(new Visitor(metadata, symbolAllocator), UnaliasContext.empty()).getRoot();
    }

    private static class Visitor
            extends PlanVisitor<PlanAndMappings, UnaliasContext>
    {
        private final Metadata metadata;
        private final SymbolAllocator symbolAllocator;

        public Visitor(Metadata metadata, SymbolAllocator symbolAllocator)
        {
            this.metadata = requireNonNull(metadata, "metadata is null");
            this.symbolAllocator = requireNonNull(symbolAllocator, "symbolAllocator is null");
        }

        @Override
        protected PlanAndMappings visitPlan(PlanNode node, UnaliasContext context)
        {
            throw new UnsupportedOperationException("Unsupported plan node " + node.getClass().getSimpleName());
        }

        @Override
        public PlanAndMappings visitAggregation(AggregationNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            AggregationNode rewrittenAggregation = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenAggregation, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitGroupId(GroupIdNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            GroupIdNode rewrittenGroupId = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenGroupId, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitExplainAnalyze(ExplainAnalyzeNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            Symbol newOutputSymbol = mapper.map(node.getOutputSymbol());

            return new PlanAndMappings(
                    new ExplainAnalyzeNode(node.getId(), rewrittenSource.getRoot(), newOutputSymbol, node.isVerbose()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitMarkDistinct(MarkDistinctNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            Symbol newMarkerSymbol = mapper.map(node.getMarkerSymbol());
            List<Symbol> newDistinctSymbols = mapper.mapAndDistinct(node.getDistinctSymbols());
            Optional<Symbol> newHashSymbol = node.getHashSymbol().map(mapper::map);

            return new PlanAndMappings(
                    new MarkDistinctNode(
                            node.getId(),
                            rewrittenSource.getRoot(),
                            newMarkerSymbol,
                            newDistinctSymbols,
                            newHashSymbol),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitUnnest(UnnestNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            List<Symbol> newReplicateSymbols = mapper.mapAndDistinct(node.getReplicateSymbols());

            ImmutableList.Builder<UnnestNode.Mapping> newMappings = ImmutableList.builder();
            for (UnnestNode.Mapping unnestMapping : node.getMappings()) {
                newMappings.add(new UnnestNode.Mapping(mapper.map(unnestMapping.getInput()), mapper.map(unnestMapping.getOutputs())));
            }

            Optional<Symbol> newOrdinalitySymbol = node.getOrdinalitySymbol().map(mapper::map);
            Optional<Expression> newFilter = node.getFilter().map(mapper::map);

            return new PlanAndMappings(
                    new UnnestNode(
                            node.getId(),
                            rewrittenSource.getRoot(),
                            newReplicateSymbols,
                            newMappings.build(),
                            newOrdinalitySymbol,
                            node.getJoinType(),
                            newFilter),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitWindow(WindowNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            WindowNode rewrittenWindow = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenWindow, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitTableScan(TableScanNode node, UnaliasContext context)
        {
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            List<Symbol> newOutputs = mapper.map(node.getOutputSymbols());

            Map<Symbol, ColumnHandle> newAssignments = new HashMap<>();
            node.getAssignments().forEach((symbol, handle) -> {
                newAssignments.put(mapper.map(symbol), handle);
            });

            return new PlanAndMappings(
                    new TableScanNode(node.getId(), node.getTable(), newOutputs, newAssignments, node.getEnforcedConstraint()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitExchange(ExchangeNode node, UnaliasContext context)
        {
            ImmutableList.Builder<PlanNode> rewrittenChildren = ImmutableList.builder();
            ImmutableList.Builder<List<Symbol>> rewrittenInputsBuilder = ImmutableList.builder();

            // rewrite child and map corresponding input list accordingly to the child's mapping
            for (int i = 0; i < node.getSources().size(); i++) {
                PlanAndMappings rewrittenChild = node.getSources().get(i).accept(this, context);
                rewrittenChildren.add(rewrittenChild.getRoot());
                AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenChild.getSymbolMappings(), context.getForbiddenSymbols());
                rewrittenInputsBuilder.add(mapper.map(node.getInputs().get(i)));
            }
            List<List<Symbol>> rewrittenInputs = rewrittenInputsBuilder.build();

            // canonicalize ExchangeNode outputs
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());
            List<Symbol> rewrittenOutputs = mapper.map(node.getOutputSymbols());

            // sanity check: assert that duplicate outputs result from same inputs
            Map<Symbol, List<Symbol>> outputsToInputs = new HashMap<>();
            for (int i = 0; i < rewrittenOutputs.size(); i++) {
                ImmutableList.Builder<Symbol> inputsBuilder = ImmutableList.builder();
                for (List<Symbol> inputs : rewrittenInputs) {
                    inputsBuilder.add(inputs.get(i));
                }
                List<Symbol> inputs = inputsBuilder.build();
                List<Symbol> previous = outputsToInputs.put(rewrittenOutputs.get(i), inputs);
                checkState(previous == null || inputs.equals(previous), "different inputs mapped to the same output symbol");
            }

            // derive new mappings for ExchangeNode output symbols
            Map<Symbol, Symbol> newMapping = new HashMap<>();

            // 1. for a single ExchangeNode source, map outputs to inputs
            if (rewrittenInputs.size() == 1) {
                for (int i = 0; i < rewrittenOutputs.size(); i++) {
                    Symbol output = rewrittenOutputs.get(i);
                    Symbol input = rewrittenInputs.get(0).get(i);
                    if (!output.equals(input)) {
                        newMapping.put(output, input);
                    }
                }
            }

            // 2. for multiple ExchangeNode sources, if different output symbols result from the same lists of canonical input symbols, map all those outputs to the same symbol
            Map<List<Symbol>, Symbol> inputsToOutputs = new HashMap<>();
            for (int i = 0; i < rewrittenOutputs.size(); i++) {
                ImmutableList.Builder<Symbol> inputsBuilder = ImmutableList.builder();
                for (List<Symbol> inputs : rewrittenInputs) {
                    inputsBuilder.add(inputs.get(i));
                }
                List<Symbol> inputs = inputsBuilder.build();
                Symbol previous = inputsToOutputs.get(inputs);
                if (previous == null || rewrittenOutputs.get(i).equals(previous)) {
                    inputsToOutputs.put(inputs, rewrittenOutputs.get(i));
                }
                else {
                    newMapping.put(rewrittenOutputs.get(i), previous);
                }
            }

            Map<Symbol, Symbol> outputMapping = new HashMap<>();
            outputMapping.putAll(mapper.getMapping());
            outputMapping.putAll(newMapping);

            mapper = new AliasingSymbolMapper(symbolAllocator, outputMapping, context.getForbiddenSymbols());

            // deduplicate outputs and prune input symbols lists accordingly
            List<List<Symbol>> newInputs = new ArrayList<>();
            for (int i = 0; i < node.getInputs().size(); i++) {
                newInputs.add(new ArrayList<>());
            }
            ImmutableList.Builder<Symbol> newOutputs = ImmutableList.builder();
            Set<Symbol> addedOutputs = new HashSet<>();
            for (int i = 0; i < rewrittenOutputs.size(); i++) {
                Symbol output = mapper.map(rewrittenOutputs.get(i));
                if (addedOutputs.add(output)) {
                    newOutputs.add(output);
                    for (int j = 0; j < rewrittenInputs.size(); j++) {
                        newInputs.get(j).add(rewrittenInputs.get(j).get(i));
                    }
                }
            }

            // rewrite PartitioningScheme
            PartitioningScheme newPartitioningScheme = mapper.map(node.getPartitioningScheme(), newOutputs.build());

            // rewrite OrderingScheme
            Optional<OrderingScheme> newOrderingScheme = node.getOrderingScheme().map(mapper::map);

            return new PlanAndMappings(
                    new ExchangeNode(
                            node.getId(),
                            node.getType(),
                            node.getScope(),
                            newPartitioningScheme,
                            rewrittenChildren.build(),
                            newInputs,
                            newOrderingScheme),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitRemoteSource(RemoteSourceNode node, UnaliasContext context)
        {
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            List<Symbol> newOutputs = mapper.mapAndDistinct(node.getOutputSymbols());
            Optional<OrderingScheme> newOrderingScheme = node.getOrderingScheme().map(mapper::map);

            return new PlanAndMappings(
                    new RemoteSourceNode(
                            node.getId(),
                            node.getSourceFragmentIds(),
                            newOutputs,
                            newOrderingScheme,
                            node.getExchangeType()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitOffset(OffsetNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);

            return new PlanAndMappings(
                    node.replaceChildren(ImmutableList.of(rewrittenSource.getRoot())),
                    rewrittenSource.getSymbolMappings());
        }

        @Override
        public PlanAndMappings visitLimit(LimitNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            LimitNode rewrittenLimit = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenLimit, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitDistinctLimit(DistinctLimitNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            DistinctLimitNode rewrittenDistinctLimit = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenDistinctLimit, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitSample(SampleNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);

            return new PlanAndMappings(
                    node.replaceChildren(ImmutableList.of(rewrittenSource.getRoot())),
                    rewrittenSource.getSymbolMappings());
        }

        @Override
        public PlanAndMappings visitValues(ValuesNode node, UnaliasContext context)
        {
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            List<List<Expression>> newRows = node.getRows().stream()
                    .map(row -> row.stream()
                            .map(mapper::map)
                            .collect(toImmutableList()))
                    .collect(toImmutableList());

            List<Symbol> newOutputSymbols = mapper.mapAndDistinct(node.getOutputSymbols());
            checkState(node.getOutputSymbols().size() == newOutputSymbols.size(), "Values output symbols were pruned");

            return new PlanAndMappings(
                    new ValuesNode(node.getId(), newOutputSymbols, newRows),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitTableDelete(TableDeleteNode node, UnaliasContext context)
        {
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            Symbol newOutput = mapper.map(node.getOutput());

            return new PlanAndMappings(
                    new TableDeleteNode(node.getId(), node.getTarget(), newOutput),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitDelete(DeleteNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            Symbol newRowId = mapper.map(node.getRowId());
            List<Symbol> newOutputs = mapper.map(node.getOutputSymbols());

            return new PlanAndMappings(
                    new DeleteNode(
                            node.getId(),
                            rewrittenSource.getRoot(),
                            node.getTarget(),
                            newRowId,
                            newOutputs),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitStatisticsWriterNode(StatisticsWriterNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            StatisticsWriterNode rewrittenStatisticsWriter = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenStatisticsWriter, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitTableWriter(TableWriterNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            TableWriterNode rewrittenTableWriter = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenTableWriter, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitTableFinish(TableFinishNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            TableFinishNode rewrittenTableFinish = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenTableFinish, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitRowNumber(RowNumberNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            RowNumberNode rewrittenRowNumber = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenRowNumber, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitTopNRowNumber(TopNRowNumberNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            TopNRowNumberNode rewrittenTopNRowNumber = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenTopNRowNumber, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitTopN(TopNNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            TopNNode rewrittenTopN = mapper.map(node, rewrittenSource.getRoot());

            return new PlanAndMappings(rewrittenTopN, mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitSort(SortNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            OrderingScheme newOrderingScheme = mapper.map(node.getOrderingScheme());

            return new PlanAndMappings(
                    new SortNode(node.getId(), rewrittenSource.getRoot(), newOrderingScheme, node.isPartial()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitFilter(FilterNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            Expression newPredicate = mapper.map(node.getPredicate());

            return new PlanAndMappings(
                    new FilterNode(node.getId(), rewrittenSource.getRoot(), newPredicate),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitProject(ProjectNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            // canonicalize ProjectNode assignments
            ImmutableList.Builder<Map.Entry<Symbol, Expression>> builder = ImmutableList.builder();
            for (Map.Entry<Symbol, Expression> assignment : node.getAssignments().entrySet()) {
                builder.add(new SimpleEntry<>(mapper.map(assignment.getKey()), mapper.map(assignment.getValue())));
            }
            List<Map.Entry<Symbol, Expression>> rewrittenAssignments = builder.build();

            // deduplicate assignments
            Map<Symbol, Expression> deduplicateAssignments = new HashMap<>();
            for (Map.Entry<Symbol, Expression> assignment : rewrittenAssignments) {
                Expression previous = deduplicateAssignments.put(assignment.getKey(), assignment.getValue());
                checkState(previous == null || assignment.getValue().equals(previous), "different expressions projected to the same symbol");
            }

            // derive new mappings for ProjectNode output symbols
            Map<Symbol, Symbol> newMapping = new HashMap<>();
            Map<Expression, Symbol> inputsToOutputs = new HashMap<>();
            for (Map.Entry<Symbol, Expression> assignment : deduplicateAssignments.entrySet()) {
                Expression expression = assignment.getValue();
                // 1. for trivial symbol projection, map output symbol to input symbol
                if (expression instanceof SymbolReference) {
                    Symbol value = Symbol.from(expression);
                    if (!assignment.getKey().equals(value)) {
                        newMapping.put(assignment.getKey(), value);
                    }
                }
                // 2. map same deterministic expressions within a projection into the same symbol
                // omit NullLiterals since those have ambiguous types
                else if (DeterminismEvaluator.isDeterministic(expression, metadata) && !(expression instanceof NullLiteral)) {
                    Symbol previous = inputsToOutputs.get(expression);
                    if (previous == null || assignment.getKey().equals(previous)) {
                        inputsToOutputs.put(expression, assignment.getKey());
                    }
                    else {
                        newMapping.put(assignment.getKey(), previous);
                    }
                }
            }

            Map<Symbol, Symbol> outputMapping = new HashMap<>();
            outputMapping.putAll(mapper.getMapping());
            outputMapping.putAll(newMapping);

            mapper = new AliasingSymbolMapper(symbolAllocator, outputMapping, context.getForbiddenSymbols());

            // build new Assignments with canonical outputs
            // duplicate entries will be removed by the Builder
            Assignments.Builder newAssignments = Assignments.builder();
            for (Map.Entry<Symbol, Expression> assignment : deduplicateAssignments.entrySet()) {
                newAssignments.put(mapper.map(assignment.getKey()), assignment.getValue());
            }

            return new PlanAndMappings(
                    new ProjectNode(node.getId(), rewrittenSource.getRoot(), newAssignments.build()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitOutput(OutputNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            List<Symbol> newOutputs = mapper.map(node.getOutputSymbols());

            return new PlanAndMappings(
                    new OutputNode(node.getId(), rewrittenSource.getRoot(), node.getColumnNames(), newOutputs),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitEnforceSingleRow(EnforceSingleRowNode node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);

            return new PlanAndMappings(
                    node.replaceChildren(ImmutableList.of(rewrittenSource.getRoot())),
                    rewrittenSource.getSymbolMappings());
        }

        @Override
        public PlanAndMappings visitAssignUniqueId(AssignUniqueId node, UnaliasContext context)
        {
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenSource.getSymbolMappings(), context.getForbiddenSymbols());

            Symbol newUnique = mapper.map(node.getIdColumn());

            return new PlanAndMappings(
                    new AssignUniqueId(node.getId(), rewrittenSource.getRoot(), newUnique),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitApply(ApplyNode node, UnaliasContext context)
        {
            // it is assumed that apart from correlation (and possibly outer correlation), symbols are distinct between Input and Subquery
            // rewrite Input
            PlanAndMappings rewrittenInput = node.getInput().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenInput.getSymbolMappings(), context.getForbiddenSymbols());

            // rewrite correlation with mapping from Input
            List<Symbol> rewrittenCorrelation = mapper.mapAndDistinct(node.getCorrelation());

            // extract new mappings for correlation symbols to apply in Subquery
            Set<Symbol> correlationSymbols = ImmutableSet.copyOf(node.getCorrelation());
            Map<Symbol, Symbol> correlationMapping = mapper.getMapping().entrySet().stream()
                    .filter(mapping -> correlationSymbols.contains(mapping.getKey()))
                    .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

            Map<Symbol, Symbol> mappingForSubquery = new HashMap<>();
            mappingForSubquery.putAll(context.getCorrelationMapping());
            mappingForSubquery.putAll(correlationMapping);

            // rewrite Subquery
            PlanAndMappings rewrittenSubquery = node.getSubquery().accept(this, new UnaliasContext(mappingForSubquery, context.getForbiddenSymbols()));

            // unify mappings from Input and Subquery to rewrite Subquery assignments
            Map<Symbol, Symbol> resultMapping = new HashMap<>();
            resultMapping.putAll(rewrittenInput.getSymbolMappings());
            resultMapping.putAll(rewrittenSubquery.getSymbolMappings());
            mapper = new AliasingSymbolMapper(symbolAllocator, resultMapping, context.getForbiddenSymbols());

            ImmutableList.Builder<Map.Entry<Symbol, Expression>> builder = ImmutableList.builder();
            for (Map.Entry<Symbol, Expression> assignment : node.getSubqueryAssignments().entrySet()) {
                builder.add(new SimpleEntry<>(mapper.map(assignment.getKey()), mapper.map(assignment.getValue())));
            }
            List<Map.Entry<Symbol, Expression>> rewrittenAssignments = builder.build();

            // deduplicate assignments
            Map<Symbol, Expression> deduplicateAssignments = new HashMap<>();
            for (Map.Entry<Symbol, Expression> assignment : rewrittenAssignments) {
                Expression previous = deduplicateAssignments.put(assignment.getKey(), assignment.getValue());
                checkState(previous == null || assignment.getValue().equals(previous), "different expressions assigned to the same symbol");
            }

            // derive new mappings for Subquery assignments outputs
            Map<Expression, Symbol> inputsToOutputs = new HashMap<>();
            Map<Symbol, Symbol> newMapping = new HashMap<>();
            for (Map.Entry<Symbol, Expression> assignment : deduplicateAssignments.entrySet()) {
                Expression expression = assignment.getValue();
                // 1. for trivial symbol projection, map output symbol to input symbol
                if (expression instanceof SymbolReference) {
                    Symbol value = Symbol.from(expression);
                    if (!assignment.getKey().equals(value)) {
                        newMapping.put(assignment.getKey(), value);
                    }
                }
                // 2. map same deterministic expressions within a projection into the same symbol
                // omit NullLiterals since those have ambiguous types
                else if (DeterminismEvaluator.isDeterministic(expression, metadata) && !(expression instanceof NullLiteral)) {
                    Symbol previous = inputsToOutputs.get(expression);
                    if (previous == null || assignment.getKey().equals(previous)) {
                        inputsToOutputs.put(expression, assignment.getKey());
                    }
                    else {
                        newMapping.put(assignment.getKey(), previous);
                    }
                }
            }

            Map<Symbol, Symbol> assignmentsOutputMapping = new HashMap<>();
            assignmentsOutputMapping.putAll(mapper.getMapping());
            assignmentsOutputMapping.putAll(newMapping);

            mapper = new AliasingSymbolMapper(symbolAllocator, assignmentsOutputMapping, context.getForbiddenSymbols());

            // build new Assignments with canonical outputs
            // duplicate entries will be removed by the Builder
            Assignments.Builder newAssignments = Assignments.builder();
            for (Map.Entry<Symbol, Expression> assignment : deduplicateAssignments.entrySet()) {
                newAssignments.put(mapper.map(assignment.getKey()), assignment.getValue());
            }

            return new PlanAndMappings(
                    new ApplyNode(node.getId(), rewrittenInput.getRoot(), rewrittenSubquery.getRoot(), newAssignments.build(), rewrittenCorrelation, node.getOriginSubquery()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitCorrelatedJoin(CorrelatedJoinNode node, UnaliasContext context)
        {
            // it is assumed that apart from correlation (and possibly outer correlation), symbols are distinct between left and right CorrelatedJoin source
            // rewrite Input
            PlanAndMappings rewrittenInput = node.getInput().accept(this, context);
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, rewrittenInput.getSymbolMappings(), context.getForbiddenSymbols());

            // rewrite correlation with mapping from Input
            List<Symbol> rewrittenCorrelation = mapper.mapAndDistinct(node.getCorrelation());

            // extract new mappings for correlation symbols to apply in Subquery
            Set<Symbol> correlationSymbols = ImmutableSet.copyOf(node.getCorrelation());
            Map<Symbol, Symbol> correlationMapping = mapper.getMapping().entrySet().stream()
                    .filter(mapping -> correlationSymbols.contains(mapping.getKey()))
                    .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

            Map<Symbol, Symbol> mappingForSubquery = new HashMap<>();
            mappingForSubquery.putAll(context.getCorrelationMapping());
            mappingForSubquery.putAll(correlationMapping);

            // rewrite Subquery
            PlanAndMappings rewrittenSubquery = node.getSubquery().accept(this, new UnaliasContext(mappingForSubquery, context.getForbiddenSymbols()));

            // unify mappings from Input and Subquery
            Map<Symbol, Symbol> resultMapping = new HashMap<>();
            resultMapping.putAll(rewrittenInput.getSymbolMappings());
            resultMapping.putAll(rewrittenSubquery.getSymbolMappings());

            // rewrite filter with unified mapping
            mapper = new AliasingSymbolMapper(symbolAllocator, resultMapping, context.getForbiddenSymbols());
            Expression newFilter = mapper.map(node.getFilter());

            return new PlanAndMappings(
                    new CorrelatedJoinNode(node.getId(), rewrittenInput.getRoot(), rewrittenSubquery.getRoot(), rewrittenCorrelation, node.getType(), newFilter, node.getOriginSubquery()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitJoin(JoinNode node, UnaliasContext context)
        {
            // it is assumed that symbols are distinct between left and right join source. Only symbols from outer correlation might be the exception
            PlanAndMappings rewrittenLeft = node.getLeft().accept(this, context);
            PlanAndMappings rewrittenRight = node.getRight().accept(this, context);

            // unify mappings from left and right join source
            Map<Symbol, Symbol> unifiedMapping = new HashMap<>();
            unifiedMapping.putAll(rewrittenLeft.getSymbolMappings());
            unifiedMapping.putAll(rewrittenRight.getSymbolMappings());

            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, unifiedMapping, context.getForbiddenSymbols());

            ImmutableList.Builder<JoinNode.EquiJoinClause> builder = ImmutableList.builder();
            for (JoinNode.EquiJoinClause clause : node.getCriteria()) {
                builder.add(new JoinNode.EquiJoinClause(mapper.map(clause.getLeft()), mapper.map(clause.getRight())));
            }
            List<JoinNode.EquiJoinClause> newCriteria = builder.build();

            Optional<Expression> newFilter = node.getFilter().map(mapper::map);
            Optional<Symbol> newLeftHashSymbol = node.getLeftHashSymbol().map(mapper::map);
            Optional<Symbol> newRightHashSymbol = node.getRightHashSymbol().map(mapper::map);

            // rewrite dynamic filters
            Set<Symbol> added = new HashSet<>();
            ImmutableMap.Builder<String, Symbol> filtersBuilder = ImmutableMap.builder();
            for (Map.Entry<String, Symbol> entry : node.getDynamicFilters().entrySet()) {
                Symbol canonical = mapper.map(entry.getValue());
                if (added.add(canonical)) {
                    filtersBuilder.put(entry.getKey(), canonical);
                }
            }
            Map<String, Symbol> newDynamicFilters = filtersBuilder.build();

            // derive new mappings from inner join equi criteria
            Map<Symbol, Symbol> newMapping = new HashMap<>();
            if (node.getType() == INNER) {
                newCriteria.stream()
                        // Map right equi-condition symbol to left symbol. This helps to
                        // reuse join node partitioning better as partitioning properties are
                        // only derived from probe side symbols
                        .forEach(clause -> newMapping.put(clause.getRight(), clause.getLeft()));
            }

            Map<Symbol, Symbol> outputMapping = new HashMap<>();
            outputMapping.putAll(mapper.getMapping());
            outputMapping.putAll(newMapping);

            mapper = new AliasingSymbolMapper(symbolAllocator, outputMapping, context.getForbiddenSymbols());
            List<Symbol> canonicalOutputs = mapper.mapAndDistinct(node.getOutputSymbols());
            List<Symbol> newLeftOutputSymbols = canonicalOutputs.stream()
                    .filter(rewrittenLeft.getRoot().getOutputSymbols()::contains)
                    .collect(toImmutableList());
            List<Symbol> newRightOutputSymbols = canonicalOutputs.stream()
                    .filter(rewrittenRight.getRoot().getOutputSymbols()::contains)
                    .collect(toImmutableList());

            return new PlanAndMappings(
                    new JoinNode(
                            node.getId(),
                            node.getType(),
                            rewrittenLeft.getRoot(),
                            rewrittenRight.getRoot(),
                            newCriteria,
                            newLeftOutputSymbols,
                            newRightOutputSymbols,
                            newFilter,
                            newLeftHashSymbol,
                            newRightHashSymbol,
                            node.getDistributionType(),
                            node.isSpillable(),
                            newDynamicFilters,
                            node.getReorderJoinStatsAndCost()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitSemiJoin(SemiJoinNode node, UnaliasContext context)
        {
            // it is assumed that symbols are distinct between SemiJoin source and filtering source. Only symbols from outer correlation might be the exception
            PlanAndMappings rewrittenSource = node.getSource().accept(this, context);
            PlanAndMappings rewrittenFilteringSource = node.getFilteringSource().accept(this, context);

            Map<Symbol, Symbol> outputMapping = new HashMap<>();
            outputMapping.putAll(rewrittenSource.getSymbolMappings());
            outputMapping.putAll(rewrittenFilteringSource.getSymbolMappings());

            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, outputMapping, context.getForbiddenSymbols());

            Symbol newSourceJoinSymbol = mapper.map(node.getSourceJoinSymbol());
            Symbol newFilteringSourceJoinSymbol = mapper.map(node.getFilteringSourceJoinSymbol());
            Symbol newSemiJoinOutput = mapper.map(node.getSemiJoinOutput());
            Optional<Symbol> newSourceHashSymbol = node.getSourceHashSymbol().map(mapper::map);
            Optional<Symbol> newFilteringSourceHashSymbol = node.getFilteringSourceHashSymbol().map(mapper::map);

            return new PlanAndMappings(
                    new SemiJoinNode(
                            node.getId(),
                            rewrittenSource.getRoot(),
                            rewrittenFilteringSource.getRoot(),
                            newSourceJoinSymbol,
                            newFilteringSourceJoinSymbol,
                            newSemiJoinOutput,
                            newSourceHashSymbol,
                            newFilteringSourceHashSymbol,
                            node.getDistributionType()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitSpatialJoin(SpatialJoinNode node, UnaliasContext context)
        {
            // it is assumed that symbols are distinct between left and right SpatialJoin source. Only symbols from outer correlation might be the exception
            PlanAndMappings rewrittenLeft = node.getLeft().accept(this, context);
            PlanAndMappings rewrittenRight = node.getRight().accept(this, context);

            Map<Symbol, Symbol> outputMapping = new HashMap<>();
            outputMapping.putAll(rewrittenLeft.getSymbolMappings());
            outputMapping.putAll(rewrittenRight.getSymbolMappings());

            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, outputMapping, context.getForbiddenSymbols());

            List<Symbol> newOutputSymbols = mapper.mapAndDistinct(node.getOutputSymbols());
            Expression newFilter = mapper.map(node.getFilter());
            Optional<Symbol> newLeftPartitionSymbol = node.getLeftPartitionSymbol().map(mapper::map);
            Optional<Symbol> newRightPartitionSymbol = node.getRightPartitionSymbol().map(mapper::map);

            return new PlanAndMappings(
                    new SpatialJoinNode(node.getId(), node.getType(), rewrittenLeft.getRoot(), rewrittenRight.getRoot(), newOutputSymbols, newFilter, newLeftPartitionSymbol, newRightPartitionSymbol, node.getKdbTree()),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitIndexJoin(IndexJoinNode node, UnaliasContext context)
        {
            // it is assumed that symbols are distinct between probeSource and indexSource. Only symbols from outer correlation might be the exception
            PlanAndMappings rewrittenProbe = node.getProbeSource().accept(this, context);
            PlanAndMappings rewrittenIndex = node.getIndexSource().accept(this, context);

            Map<Symbol, Symbol> outputMapping = new HashMap<>();
            outputMapping.putAll(rewrittenProbe.getSymbolMappings());
            outputMapping.putAll(rewrittenIndex.getSymbolMappings());

            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, outputMapping, context.getForbiddenSymbols());

            // canonicalize index join criteria
            ImmutableList.Builder<IndexJoinNode.EquiJoinClause> builder = ImmutableList.builder();
            for (IndexJoinNode.EquiJoinClause clause : node.getCriteria()) {
                builder.add(new IndexJoinNode.EquiJoinClause(mapper.map(clause.getProbe()), mapper.map(clause.getIndex())));
            }
            List<IndexJoinNode.EquiJoinClause> newEquiCriteria = builder.build();

            Optional<Symbol> newProbeHashSymbol = node.getProbeHashSymbol().map(mapper::map);
            Optional<Symbol> newIndexHashSymbol = node.getIndexHashSymbol().map(mapper::map);

            return new PlanAndMappings(
                    new IndexJoinNode(node.getId(), node.getType(), rewrittenProbe.getRoot(), rewrittenIndex.getRoot(), newEquiCriteria, newProbeHashSymbol, newIndexHashSymbol),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitIndexSource(IndexSourceNode node, UnaliasContext context)
        {
            AliasingSymbolMapper mapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            Set<Symbol> newLookupSymbols = node.getLookupSymbols().stream()
                    .map(mapper::map)
                    .collect(toImmutableSet());
            List<Symbol> newOutputSymbols = mapper.mapAndDistinct(node.getOutputSymbols());

            Map<Symbol, ColumnHandle> newAssignments = new HashMap<>();
            node.getAssignments().entrySet().stream()
                    .forEach(assignment -> newAssignments.put(mapper.map(assignment.getKey()), assignment.getValue()));

            return new PlanAndMappings(
                    new IndexSourceNode(node.getId(), node.getIndexHandle(), node.getTableHandle(), newLookupSymbols, newOutputSymbols, newAssignments),
                    mapper.getMapping());
        }

        @Override
        public PlanAndMappings visitUnion(UnionNode node, UnaliasContext context)
        {
            List<PlanAndMappings> rewrittenSources = node.getSources().stream()
                    .map(source -> source.accept(this, context))
                    .collect(toImmutableList());

            List<AliasingSymbolMapper> inputMappers = rewrittenSources.stream()
                    .map(source -> new AliasingSymbolMapper(symbolAllocator, source.getSymbolMappings(), context.getForbiddenSymbols()))
                    .collect(toImmutableList());

            AliasingSymbolMapper outputMapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            ListMultimap<Symbol, Symbol> newOutputToInputs = rewriteOutputToInputsMap(node.getSymbolMapping(), outputMapper, inputMappers);
            List<Symbol> newOutputs = outputMapper.mapAndDistinct(node.getOutputSymbols());

            return new PlanAndMappings(
                    new UnionNode(
                            node.getId(),
                            rewrittenSources.stream()
                                    .map(PlanAndMappings::getRoot)
                                    .collect(toImmutableList()),
                            newOutputToInputs,
                            newOutputs),
                    outputMapper.getMapping());
        }

        @Override
        public PlanAndMappings visitIntersect(IntersectNode node, UnaliasContext context)
        {
            List<PlanAndMappings> rewrittenSources = node.getSources().stream()
                    .map(source -> source.accept(this, context))
                    .collect(toImmutableList());

            List<AliasingSymbolMapper> inputMappers = rewrittenSources.stream()
                    .map(source -> new AliasingSymbolMapper(symbolAllocator, source.getSymbolMappings(), context.getForbiddenSymbols()))
                    .collect(toImmutableList());

            AliasingSymbolMapper outputMapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            ListMultimap<Symbol, Symbol> newOutputToInputs = rewriteOutputToInputsMap(node.getSymbolMapping(), outputMapper, inputMappers);
            List<Symbol> newOutputs = outputMapper.mapAndDistinct(node.getOutputSymbols());

            return new PlanAndMappings(
                    new IntersectNode(
                            node.getId(),
                            rewrittenSources.stream()
                                    .map(PlanAndMappings::getRoot)
                                    .collect(toImmutableList()),
                            newOutputToInputs,
                            newOutputs),
                    outputMapper.getMapping());
        }

        @Override
        public PlanAndMappings visitExcept(ExceptNode node, UnaliasContext context)
        {
            List<PlanAndMappings> rewrittenSources = node.getSources().stream()
                    .map(source -> source.accept(this, context))
                    .collect(toImmutableList());

            List<AliasingSymbolMapper> inputMappers = rewrittenSources.stream()
                    .map(source -> new AliasingSymbolMapper(symbolAllocator, source.getSymbolMappings(), context.getForbiddenSymbols()))
                    .collect(toImmutableList());

            AliasingSymbolMapper outputMapper = new AliasingSymbolMapper(symbolAllocator, context.getCorrelationMapping(), context.getForbiddenSymbols());

            ListMultimap<Symbol, Symbol> newOutputToInputs = rewriteOutputToInputsMap(node.getSymbolMapping(), outputMapper, inputMappers);
            List<Symbol> newOutputs = outputMapper.mapAndDistinct(node.getOutputSymbols());

            return new PlanAndMappings(
                    new ExceptNode(
                            node.getId(),
                            rewrittenSources.stream()
                                    .map(PlanAndMappings::getRoot)
                                    .collect(toImmutableList()),
                            newOutputToInputs,
                            newOutputs),
                    outputMapper.getMapping());
        }

        private ListMultimap<Symbol, Symbol> rewriteOutputToInputsMap(ListMultimap<Symbol, Symbol> oldMapping, AliasingSymbolMapper outputMapper, List<AliasingSymbolMapper> inputMappers)
        {
            ImmutableListMultimap.Builder<Symbol, Symbol> newMappingBuilder = ImmutableListMultimap.builder();
            Set<Symbol> addedSymbols = new HashSet<>();
            for (Map.Entry<Symbol, Collection<Symbol>> entry : oldMapping.asMap().entrySet()) {
                Symbol rewrittenOutput = outputMapper.map(entry.getKey());
                if (addedSymbols.add(rewrittenOutput)) {
                    List<Symbol> inputs = ImmutableList.copyOf(entry.getValue());
                    ImmutableList.Builder<Symbol> rewrittenInputs = ImmutableList.builder();
                    for (int i = 0; i < inputs.size(); i++) {
                        rewrittenInputs.add(inputMappers.get(i).map(inputs.get(i)));
                    }
                    newMappingBuilder.putAll(rewrittenOutput, rewrittenInputs.build());
                }
            }
            return newMappingBuilder.build();
        }

        /*private static ImmutableList.Builder<PlanNode> rewriteSources(SetOperationNode node, SimplePlanRewriter.RewriteContext<Void> context)
        {
            ImmutableList.Builder<PlanNode> preRewrittenSources = ImmutableList.builder();
            for (PlanNode source : node.getSources()) {
                preRewrittenSources.add(context.rewrite(source));
            }
            ImmutableList.Builder<PlanNode> rewrittenSources = ImmutableList.builder();
            for (PlanNode source : preRewrittenSources.build()) {
                rewrittenSources.add(context.rewrite(source));
            }
            return rewrittenSources;
        }

        private ListMultimap<Symbol, Symbol> canonicalizeSetOperationSymbolMap(ListMultimap<Symbol, Symbol> setOperationSymbolMap)
        {
            ImmutableListMultimap.Builder<Symbol, Symbol> builder = ImmutableListMultimap.builder();
            Set<Symbol> addedSymbols = new HashSet<>();
            for (Map.Entry<Symbol, Collection<Symbol>> entry : setOperationSymbolMap.asMap().entrySet()) {
                Symbol canonicalOutputSymbol = canonicalize(entry.getKey());
                if (addedSymbols.add(canonicalOutputSymbol)) {
                    builder.putAll(
                            canonicalOutputSymbol,
                            entry.getValue().stream()
                                    .map(this::canonicalize)
                                    .collect(Collectors.toList()));
                }
            }
            return builder.build();
        }*/

        // TODO support all other nodes (for the sake of forbidden symbols)
        // TODO check where else new mappings were recorded (master -> method map()) and if that was not lost...

        /*private WindowNode.Frame canonicalize(WindowNode.Frame frame)
        {
            return new WindowNode.Frame(
                    frame.getType(),
                    frame.getStartType(),
                    canonicalize(frame.getStartValue()),
                    frame.getEndType(),
                    canonicalize(frame.getEndValue()),
                    frame.getOriginalStartValue(),
                    frame.getOriginalEndValue());
        }*/

        // Return the canonical mapping for the symbol.
        // If the symbol is not allowed in this context ("forbidden") and hasn't got a mapping yet,
        // map it to a new symbol. Following occurrences of the symbol will be mapped to the new symbol.
        /*private Symbol canonicalize(Symbol symbol, Map<Symbol, Symbol> mapping, Set<Symbol> forbidden)
        {
            while (mapping.containsKey(symbol)) {
                symbol = mapping.get(symbol);
            }

            if (forbidden.contains(symbol)) {
                Symbol newSymbol = symbolAllocator.newSymbol(symbol);
                mapping.put(symbol, newSymbol);
                return newSymbol;
            }

            return symbol;
        }

        private Optional<Symbol> canonicalize(Optional<Symbol> symbol, Map<Symbol, Symbol> mapping, Set<Symbol> forbidden)
        {
            return symbol.map(oldSymbol -> canonicalize(oldSymbol, mapping, forbidden));
        }

        private List<Symbol> canonicalizeAndDistinct(List<Symbol> symbols, Map<Symbol, Symbol> mapping, Set<Symbol> forbidden)
        {
            Set<Symbol> added = new HashSet<>();
            ImmutableList.Builder<Symbol> builder = ImmutableList.builder();
            for (Symbol symbol : symbols) {
                Symbol canonical = canonicalize(symbol, mapping, forbidden);
                if (added.add(canonical)) {
                    builder.add(canonical);
                }
            }
            return builder.build();
        }*/
    }

//    private static class Rewriter
//            extends SimplePlanRewriter<Void>
//    {
        /*private final Map<Symbol, Symbol> mapping = new HashMap<>();
        private final Metadata metadata;
        private final TypeProvider types;

        private Rewriter(Metadata metadata, TypeProvider types)
        {
            this.metadata = metadata;
            this.types = types;
        }*/

        /*@Override
        public PlanNode visitAggregation(AggregationNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            //TODO: use mapper in other methods
            SymbolMapper mapper = new SymbolMapper(mapping);
            return mapper.map(node, source);
        }*/

        /*@Override
        public PlanNode visitGroupId(GroupIdNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            Map<Symbol, Symbol> newGroupingMappings = new HashMap<>();
            ImmutableList.Builder<List<Symbol>> newGroupingSets = ImmutableList.builder();

            for (List<Symbol> groupingSet : node.getGroupingSets()) {
                ImmutableList.Builder<Symbol> newGroupingSet = ImmutableList.builder();
                for (Symbol output : groupingSet) {
                    newGroupingMappings.putIfAbsent(canonicalize(output), canonicalize(node.getGroupingColumns().get(output)));
                    newGroupingSet.add(canonicalize(output));
                }
                newGroupingSets.add(newGroupingSet.build());
            }

            return new GroupIdNode(node.getId(), source, newGroupingSets.build(), newGroupingMappings, canonicalizeAndDistinct(node.getAggregationArguments()), canonicalize(node.getGroupIdSymbol()));
        }*/

        /*@Override
        public PlanNode visitExplainAnalyze(ExplainAnalyzeNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            return new ExplainAnalyzeNode(node.getId(), source, canonicalize(node.getOutputSymbol()), node.isVerbose());
        }*/

        /*@Override
        public PlanNode visitMarkDistinct(MarkDistinctNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            List<Symbol> symbols = canonicalizeAndDistinct(node.getDistinctSymbols());
            return new MarkDistinctNode(node.getId(), source, canonicalize(node.getMarkerSymbol()), symbols, canonicalize(node.getHashSymbol()));
        }*/

        /*@Override
        public PlanNode visitUnnest(UnnestNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            ImmutableList.Builder<UnnestNode.Mapping> mappings = ImmutableList.builder();

            for (UnnestNode.Mapping mapping : node.getMappings()) {
                mappings.add(new UnnestNode.Mapping(canonicalize(mapping.getInput()), mapping.getOutputs()));
            }

            return new UnnestNode(
                    node.getId(),
                    source,
                    canonicalizeAndDistinct(node.getReplicateSymbols()),
                    mappings.build(),
                    node.getOrdinalitySymbol(),
                    node.getJoinType(),
                    node.getFilter().map(this::canonicalize));
        }*/

        /*@Override
        public PlanNode visitWindow(WindowNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            ImmutableMap.Builder<Symbol, WindowNode.Function> functions = ImmutableMap.builder();
            node.getWindowFunctions().forEach((symbol, function) -> {
                ResolvedFunction resolvedFunction = function.getResolvedFunction();
                List<Expression> arguments = canonicalize(function.getArguments());
                WindowNode.Frame canonicalFrame = canonicalize(function.getFrame());

                functions.put(canonicalize(symbol), new WindowNode.Function(resolvedFunction, arguments, canonicalFrame, function.isIgnoreNulls()));
            });

            return new WindowNode(
                    node.getId(),
                    source,
                    canonicalizeAndDistinct(node.getSpecification()),
                    functions.build(),
                    canonicalize(node.getHashSymbol()),
                    canonicalize(node.getPrePartitionedInputs()),
                    node.getPreSortedOrderPrefix());
        }

        private WindowNode.Frame canonicalize(WindowNode.Frame frame)
        {
            return new WindowNode.Frame(
                    frame.getType(),
                    frame.getStartType(),
                    canonicalize(frame.getStartValue()),
                    frame.getEndType(),
                    canonicalize(frame.getEndValue()),
                    frame.getOriginalStartValue(),
                    frame.getOriginalEndValue());
        }*/

        /*@Override
        public PlanNode visitTableScan(TableScanNode node, RewriteContext<Void> context)
        {
            return node;
        }*/

        /*@Override
        public PlanNode visitExchange(ExchangeNode node, RewriteContext<Void> context)
        {
            List<PlanNode> sources = node.getSources().stream()
                    .map(context::rewrite)
                    .collect(toImmutableList());

            mapExchangeNodeSymbols(node);

            List<List<Symbol>> inputs = new ArrayList<>();
            for (int i = 0; i < node.getInputs().size(); i++) {
                inputs.add(new ArrayList<>());
            }
            Set<Symbol> addedOutputs = new HashSet<>();
            ImmutableList.Builder<Symbol> outputs = ImmutableList.builder();
            for (int symbolIndex = 0; symbolIndex < node.getOutputSymbols().size(); symbolIndex++) {
                Symbol canonicalOutput = canonicalize(node.getOutputSymbols().get(symbolIndex));
                if (addedOutputs.add(canonicalOutput)) {
                    outputs.add(canonicalOutput);
                    for (int i = 0; i < node.getInputs().size(); i++) {
                        List<Symbol> input = node.getInputs().get(i);
                        inputs.get(i).add(canonicalize(input.get(symbolIndex)));
                    }
                }
            }

            PartitioningScheme partitioningScheme = new PartitioningScheme(
                    node.getPartitioningScheme().getPartitioning().translate(this::canonicalize),
                    outputs.build(),
                    canonicalize(node.getPartitioningScheme().getHashColumn()),
                    node.getPartitioningScheme().isReplicateNullsAndAny(),
                    node.getPartitioningScheme().getBucketToPartition());

            Optional<OrderingScheme> orderingScheme = node.getOrderingScheme().map(this::canonicalizeAndDistinct);

            return new ExchangeNode(node.getId(), node.getType(), node.getScope(), partitioningScheme, sources, inputs, orderingScheme);
        }*/

        /*private void mapExchangeNodeSymbols(ExchangeNode node)
        {
            if (node.getInputs().size() == 1) {
                mapExchangeNodeOutputToInputSymbols(node);
                return;
            }

            // Mapping from list [node.getInput(0).get(symbolIndex), node.getInput(1).get(symbolIndex), ...] to node.getOutputSymbols(symbolIndex).
            // All symbols are canonical.
            Map<List<Symbol>, Symbol> inputsToOutputs = new HashMap<>();
            // Map each same list of input symbols [I1, I2, ..., In] to the same output symbol O
            for (int symbolIndex = 0; symbolIndex < node.getOutputSymbols().size(); symbolIndex++) {
                Symbol canonicalOutput = canonicalize(node.getOutputSymbols().get(symbolIndex));
                List<Symbol> canonicalInputs = canonicalizeExchangeNodeInputs(node, symbolIndex);
                Symbol output = inputsToOutputs.get(canonicalInputs);

                if (output == null || canonicalOutput.equals(output)) {
                    inputsToOutputs.put(canonicalInputs, canonicalOutput);
                }
                else {
                    map(canonicalOutput, output);
                }
            }
        }

        private void mapExchangeNodeOutputToInputSymbols(ExchangeNode node)
        {
            checkState(node.getInputs().size() == 1);

            for (int symbolIndex = 0; symbolIndex < node.getOutputSymbols().size(); symbolIndex++) {
                Symbol canonicalOutput = canonicalize(node.getOutputSymbols().get(symbolIndex));
                Symbol canonicalInput = canonicalize(node.getInputs().get(0).get(symbolIndex));

                if (!canonicalOutput.equals(canonicalInput)) {
                    map(canonicalOutput, canonicalInput);
                }
            }
        }

        private List<Symbol> canonicalizeExchangeNodeInputs(ExchangeNode node, int symbolIndex)
        {
            return node.getInputs().stream()
                    .map(input -> canonicalize(input.get(symbolIndex)))
                    .collect(toImmutableList());
        }*/

        /*@Override
        public PlanNode visitRemoteSource(RemoteSourceNode node, RewriteContext<Void> context)
        {
            return new RemoteSourceNode(
                    node.getId(),
                    node.getSourceFragmentIds(),
                    canonicalizeAndDistinct(node.getOutputSymbols()),
                    node.getOrderingScheme().map(this::canonicalizeAndDistinct),
                    node.getExchangeType());
        }*/

        /*@Override
        public PlanNode visitOffset(OffsetNode node, RewriteContext<Void> context)
        {
            return context.defaultRewrite(node);
        }*/

        /*@Override
        public PlanNode visitLimit(LimitNode node, RewriteContext<Void> context)
        {
            if (node.isWithTies()) {
                PlanNode source = context.rewrite(node.getSource());
                return new LimitNode(node.getId(), source, node.getCount(), node.getTiesResolvingScheme().map(this::canonicalizeAndDistinct), node.isPartial());
            }
            return context.defaultRewrite(node);
        }*/

        /*@Override
        public PlanNode visitDistinctLimit(DistinctLimitNode node, RewriteContext<Void> context)
        {
            return new DistinctLimitNode(node.getId(), context.rewrite(node.getSource()), node.getLimit(), node.isPartial(), canonicalizeAndDistinct(node.getDistinctSymbols()), canonicalize(node.getHashSymbol()));
        }*/

        /*@Override
        public PlanNode visitSample(SampleNode node, RewriteContext<Void> context)
        {
            return new SampleNode(node.getId(), context.rewrite(node.getSource()), node.getSampleRatio(), node.getSampleType());
        }*/

        /*@Override
        public PlanNode visitValues(ValuesNode node, RewriteContext<Void> context)
        {
            List<List<Expression>> canonicalizedRows = node.getRows().stream()
                    .map(this::canonicalize)
                    .collect(toImmutableList());
            List<Symbol> canonicalizedOutputSymbols = canonicalizeAndDistinct(node.getOutputSymbols());
            checkState(node.getOutputSymbols().size() == canonicalizedOutputSymbols.size(), "Values output symbols were pruned");
            return new ValuesNode(
                    node.getId(),
                    canonicalizedOutputSymbols,
                    canonicalizedRows);
        }*/

        /*@Override
        public PlanNode visitTableDelete(TableDeleteNode node, RewriteContext<Void> context)
        {
            return node;
        }

        @Override
        public PlanNode visitDelete(DeleteNode node, RewriteContext<Void> context)
        {
            return new DeleteNode(node.getId(), context.rewrite(node.getSource()), node.getTarget(), canonicalize(node.getRowId()), node.getOutputSymbols());
        }*/

        /*@Override
        public PlanNode visitStatisticsWriterNode(StatisticsWriterNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            SymbolMapper mapper = new SymbolMapper(mapping);
            return mapper.map(node, source);
        }*/

        /*@Override
        public PlanNode visitTableFinish(TableFinishNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            SymbolMapper mapper = new SymbolMapper(mapping);
            return mapper.map(node, source);
        }*/

        /*@Override
        public PlanNode visitRowNumber(RowNumberNode node, RewriteContext<Void> context)
        {
            return new RowNumberNode(
                    node.getId(),
                    context.rewrite(node.getSource()),
                    canonicalizeAndDistinct(node.getPartitionBy()),
                    node.isOrderSensitive(),
                    canonicalize(node.getRowNumberSymbol()),
                    node.getMaxRowCountPerPartition(),
                    canonicalize(node.getHashSymbol()));
        }*/

        /*@Override
        public PlanNode visitTopNRowNumber(TopNRowNumberNode node, RewriteContext<Void> context)
        {
            return new TopNRowNumberNode(
                    node.getId(),
                    context.rewrite(node.getSource()),
                    canonicalizeAndDistinct(node.getSpecification()),
                    canonicalize(node.getRowNumberSymbol()),
                    node.getMaxRowCountPerPartition(),
                    node.isPartial(),
                    canonicalize(node.getHashSymbol()));
        }*/

        /*@Override
        public PlanNode visitFilter(FilterNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            return new FilterNode(node.getId(), source, canonicalize(node.getPredicate()));
        }*/

        /*@Override
        public PlanNode visitProject(ProjectNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            return new ProjectNode(node.getId(), source, canonicalize(node.getAssignments()));
        }*/

        /*@Override
        public PlanNode visitOutput(OutputNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            List<Symbol> canonical = Lists.transform(node.getOutputSymbols(), this::canonicalize);
            return new OutputNode(node.getId(), source, node.getColumnNames(), canonical);
        }*/

        /*@Override
        public PlanNode visitEnforceSingleRow(EnforceSingleRowNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            return new EnforceSingleRowNode(node.getId(), source);
        }*/

        /*@Override
        public PlanNode visitAssignUniqueId(AssignUniqueId node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            return new AssignUniqueId(node.getId(), source, node.getIdColumn());
        }*/

        /*@Override
        public PlanNode visitApply(ApplyNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getInput());
            PlanNode subquery = context.rewrite(node.getSubquery());
            List<Symbol> canonicalCorrelation = Lists.transform(node.getCorrelation(), this::canonicalize);

            return new ApplyNode(node.getId(), source, subquery, canonicalize(node.getSubqueryAssignments()), canonicalCorrelation, node.getOriginSubquery());
        }*/

        /*@Override
        public PlanNode visitCorrelatedJoin(CorrelatedJoinNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getInput());
            PlanNode subquery = context.rewrite(node.getSubquery());
            List<Symbol> canonicalCorrelation = canonicalizeAndDistinct(node.getCorrelation());

            return new CorrelatedJoinNode(node.getId(), source, subquery, canonicalCorrelation, node.getType(), canonicalize(node.getFilter()), node.getOriginSubquery());
        }*/

        /*@Override
        public PlanNode visitTopN(TopNNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            SymbolMapper mapper = new SymbolMapper(mapping);
            return mapper.map(node, source, node.getId());
        }*/

        /*@Override
        public PlanNode visitSort(SortNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            return new SortNode(node.getId(), source, canonicalizeAndDistinct(node.getOrderingScheme()), node.isPartial());
        }*/

        /*@Override
        public PlanNode visitJoin(JoinNode node, RewriteContext<Void> context)
        {
            PlanNode left = context.rewrite(node.getLeft());
            PlanNode right = context.rewrite(node.getRight());

            List<JoinNode.EquiJoinClause> canonicalCriteria = canonicalizeJoinCriteria(node.getCriteria());
            Optional<Expression> canonicalFilter = node.getFilter().map(this::canonicalize);
            Optional<Symbol> canonicalLeftHashSymbol = canonicalize(node.getLeftHashSymbol());
            Optional<Symbol> canonicalRightHashSymbol = canonicalize(node.getRightHashSymbol());

            Map<String, Symbol> canonicalDynamicFilters = canonicalizeAndDistinct(node.getDynamicFilters());

            if (node.getType() == INNER) {
                canonicalCriteria.stream()
                        // Map right equi-condition symbol to left symbol. This helps to
                        // reuse join node partitioning better as partitioning properties are
                        // only derived from probe side symbols
                        .forEach(clause -> map(clause.getRight(), clause.getLeft()));
            }

            List<Symbol> canonicalOutputs = canonicalizeAndDistinct(node.getOutputSymbols());
            List<Symbol> leftOutputSymbols = canonicalOutputs.stream()
                    .filter(left.getOutputSymbols()::contains)
                    .collect(toImmutableList());
            List<Symbol> rightOutputSymbols = canonicalOutputs.stream()
                    .filter(right.getOutputSymbols()::contains)
                    .collect(toImmutableList());

            return new JoinNode(
                    node.getId(),
                    node.getType(),
                    left,
                    right,
                    canonicalCriteria,
                    leftOutputSymbols,
                    rightOutputSymbols,
                    canonicalFilter,
                    canonicalLeftHashSymbol,
                    canonicalRightHashSymbol,
                    node.getDistributionType(),
                    node.isSpillable(),
                    canonicalDynamicFilters,
                    node.getReorderJoinStatsAndCost());
        }*/

        /*@Override
        public PlanNode visitSemiJoin(SemiJoinNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            PlanNode filteringSource = context.rewrite(node.getFilteringSource());

            return new SemiJoinNode(
                    node.getId(),
                    source,
                    filteringSource,
                    canonicalize(node.getSourceJoinSymbol()),
                    canonicalize(node.getFilteringSourceJoinSymbol()),
                    canonicalize(node.getSemiJoinOutput()),
                    canonicalize(node.getSourceHashSymbol()),
                    canonicalize(node.getFilteringSourceHashSymbol()),
                    node.getDistributionType());
        }*/

        /*@Override
        public PlanNode visitSpatialJoin(SpatialJoinNode node, RewriteContext<Void> context)
        {
            PlanNode left = context.rewrite(node.getLeft());
            PlanNode right = context.rewrite(node.getRight());

            return new SpatialJoinNode(node.getId(), node.getType(), left, right, canonicalizeAndDistinct(node.getOutputSymbols()), canonicalize(node.getFilter()), canonicalize(node.getLeftPartitionSymbol()), canonicalize(node.getRightPartitionSymbol()), node.getKdbTree());
        }*/

        /*@Override
        public PlanNode visitIndexSource(IndexSourceNode node, RewriteContext<Void> context)
        {
            return new IndexSourceNode(node.getId(), node.getIndexHandle(), node.getTableHandle(), canonicalize(node.getLookupSymbols()), node.getOutputSymbols(), node.getAssignments());
        }*/

        /*@Override
        public PlanNode visitIndexJoin(IndexJoinNode node, RewriteContext<Void> context)
        {
            PlanNode probeSource = context.rewrite(node.getProbeSource());
            PlanNode indexSource = context.rewrite(node.getIndexSource());

            return new IndexJoinNode(node.getId(), node.getType(), probeSource, indexSource, canonicalizeIndexJoinCriteria(node.getCriteria()), canonicalize(node.getProbeHashSymbol()), canonicalize(node.getIndexHashSymbol()));
        }*/

        /*@Override
        public PlanNode visitUnion(UnionNode node, RewriteContext<Void> context)
        {
            return new UnionNode(node.getId(), rewriteSources(node, context).build(), canonicalizeSetOperationSymbolMap(node.getSymbolMapping()), canonicalizeAndDistinct(node.getOutputSymbols()));
        }

        @Override
        public PlanNode visitIntersect(IntersectNode node, RewriteContext<Void> context)
        {
            return new IntersectNode(node.getId(), rewriteSources(node, context).build(), canonicalizeSetOperationSymbolMap(node.getSymbolMapping()), canonicalizeAndDistinct(node.getOutputSymbols()));
        }

        @Override
        public PlanNode visitExcept(ExceptNode node, RewriteContext<Void> context)
        {
            return new ExceptNode(node.getId(), rewriteSources(node, context).build(), canonicalizeSetOperationSymbolMap(node.getSymbolMapping()), canonicalizeAndDistinct(node.getOutputSymbols()));
        }

        private static ImmutableList.Builder<PlanNode> rewriteSources(SetOperationNode node, RewriteContext<Void> context)
        {
            ImmutableList.Builder<PlanNode> preRewrittenSources = ImmutableList.builder();
            for (PlanNode source : node.getSources()) {
                preRewrittenSources.add(context.rewrite(source));
            }
            ImmutableList.Builder<PlanNode> rewrittenSources = ImmutableList.builder();
            for (PlanNode source : preRewrittenSources.build()) {
                rewrittenSources.add(context.rewrite(source));
            }
            return rewrittenSources;
        }*/

        /*@Override
        public PlanNode visitTableWriter(TableWriterNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());
            SymbolMapper mapper = new SymbolMapper(mapping);
            return mapper.map(node, source);
        }*/

        /*@Override
        protected PlanNode visitPlan(PlanNode node, RewriteContext<Void> context)
        {
            throw new UnsupportedOperationException("Unsupported plan node " + node.getClass().getSimpleName());
        }*/

        /*private void map(Symbol symbol, Symbol canonical)
        {
            Preconditions.checkArgument(!symbol.equals(canonical), "Can't map symbol to itself: %s", symbol);
            mapping.put(symbol, canonical);
        }*/

        /*private Assignments canonicalize(Assignments oldAssignments)
        {
            Map<Expression, Symbol> computedExpressions = new HashMap<>();
            Assignments.Builder assignments = Assignments.builder();
            for (Map.Entry<Symbol, Expression> entry : oldAssignments.getMap().entrySet()) {
                Expression expression = canonicalize(entry.getValue());

                if (expression instanceof SymbolReference) {
                    // Always map a trivial symbol projection
                    Symbol symbol = Symbol.from(expression);
                    if (!symbol.equals(entry.getKey())) {
                        map(entry.getKey(), symbol);
                    }
                }
                else if (DeterminismEvaluator.isDeterministic(expression, metadata) && !(expression instanceof NullLiteral)) {
                    // Try to map same deterministic expressions within a projection into the same symbol
                    // Omit NullLiterals since those have ambiguous types
                    Symbol computedSymbol = computedExpressions.get(expression);
                    if (computedSymbol == null) {
                        // If we haven't seen the expression before in this projection, record it
                        computedExpressions.put(expression, entry.getKey());
                    }
                    else {
                        // If we have seen the expression before and if it is deterministic
                        // then we can rewrite references to the current symbol in terms of the parallel computedSymbol in the projection
                        map(entry.getKey(), computedSymbol);
                    }
                }

                Symbol canonical = canonicalize(entry.getKey());
                assignments.put(canonical, expression);
            }
            return assignments.build();
        }*/

        /*private Optional<Symbol> canonicalize(Optional<Symbol> symbol)
        {
            if (symbol.isPresent()) {
                return Optional.of(canonicalize(symbol.get()));
            }
            return Optional.empty();
        }*/

        /*private Symbol canonicalize(Symbol symbol)
        {
            Symbol canonical = symbol;
            while (mapping.containsKey(canonical)) {
                canonical = mapping.get(canonical);
            }
            return canonical;
        }*/

        /*private List<Expression> canonicalize(List<Expression> values)
        {
            return values.stream()
                    .map(this::canonicalize)
                    .collect(toImmutableList());
        }*/

        /*private Expression canonicalize(Expression value)
        {
            return ExpressionTreeRewriter.rewriteWith(new ExpressionRewriter<Void>()
            {
                @Override
                public Expression rewriteSymbolReference(SymbolReference node, Void context, ExpressionTreeRewriter<Void> treeRewriter)
                {
                    Symbol canonical = canonicalize(Symbol.from(node));
                    return canonical.toSymbolReference();
                }
            }, value);
        }*/

        /*private List<Symbol> canonicalizeAndDistinct(List<Symbol> outputs)
        {
            Set<Symbol> added = new HashSet<>();
            ImmutableList.Builder<Symbol> builder = ImmutableList.builder();
            for (Symbol symbol : outputs) {
                Symbol canonical = canonicalize(symbol);
                if (added.add(canonical)) {
                    builder.add(canonical);
                }
            }
            return builder.build();
        }*/

        /*private Map<String, Symbol> canonicalizeAndDistinct(Map<String, Symbol> dynamicFilters)
        {
            Set<Symbol> added = new HashSet<>();
            ImmutableMap.Builder<String, Symbol> builder = ImmutableMap.builder();
            for (Map.Entry<String, Symbol> entry : dynamicFilters.entrySet()) {
                Symbol canonical = canonicalize(entry.getValue());
                if (added.add(canonical)) {
                    builder.put(entry.getKey(), canonical);
                }
            }
            return builder.build();
        }*/

        /*private WindowNode.Specification canonicalizeAndDistinct(WindowNode.Specification specification)
        {
            return new WindowNode.Specification(
                    canonicalizeAndDistinct(specification.getPartitionBy()),
                    specification.getOrderingScheme().map(this::canonicalizeAndDistinct));
        }*/

        /*private OrderingScheme canonicalizeAndDistinct(OrderingScheme orderingScheme)
        {
            Set<Symbol> added = new HashSet<>();
            ImmutableList.Builder<Symbol> symbols = ImmutableList.builder();
            ImmutableMap.Builder<Symbol, SortOrder> orderings = ImmutableMap.builder();
            for (Symbol symbol : orderingScheme.getOrderBy()) {
                Symbol canonical = canonicalize(symbol);
                if (added.add(canonical)) {
                    symbols.add(canonical);
                    orderings.put(canonical, orderingScheme.getOrdering(symbol));
                }
            }

            return new OrderingScheme(symbols.build(), orderings.build());
        }*/

        /*private Set<Symbol> canonicalize(Set<Symbol> symbols)
        {
            return symbols.stream()
                    .map(this::canonicalize)
                    .collect(toImmutableSet());
        }*/

        /*private List<JoinNode.EquiJoinClause> canonicalizeJoinCriteria(List<JoinNode.EquiJoinClause> criteria)
        {
            ImmutableList.Builder<JoinNode.EquiJoinClause> builder = ImmutableList.builder();
            for (JoinNode.EquiJoinClause clause : criteria) {
                builder.add(new JoinNode.EquiJoinClause(canonicalize(clause.getLeft()), canonicalize(clause.getRight())));
            }

            return builder.build();
        }*/

        /*private List<IndexJoinNode.EquiJoinClause> canonicalizeIndexJoinCriteria(List<IndexJoinNode.EquiJoinClause> criteria)
        {
            ImmutableList.Builder<IndexJoinNode.EquiJoinClause> builder = ImmutableList.builder();
            for (IndexJoinNode.EquiJoinClause clause : criteria) {
                builder.add(new IndexJoinNode.EquiJoinClause(canonicalize(clause.getProbe()), canonicalize(clause.getIndex())));
            }

            return builder.build();
        }*/

        /*private ListMultimap<Symbol, Symbol> canonicalizeSetOperationSymbolMap(ListMultimap<Symbol, Symbol> setOperationSymbolMap)
        {
            ImmutableListMultimap.Builder<Symbol, Symbol> builder = ImmutableListMultimap.builder();
            Set<Symbol> addedSymbols = new HashSet<>();
            for (Map.Entry<Symbol, Collection<Symbol>> entry : setOperationSymbolMap.asMap().entrySet()) {
                Symbol canonicalOutputSymbol = canonicalize(entry.getKey());
                if (addedSymbols.add(canonicalOutputSymbol)) {
                    builder.putAll(
                            canonicalOutputSymbol,
                            entry.getValue().stream()
                                    .map(this::canonicalize)
                                    .collect(Collectors.toList()));
                }
            }
            return builder.build();
        }*/
//    }

    private static class UnaliasContext
    {
        // Correlation mapping is a record of how correlation symbols have been mapped in the subplan which provides them.
        // All occurrences of correlation symbols within the correlated subquery must be remapped accordingly.
        // In case of nested correlation, correlationMappings has required mappings for correlation symbols from all levels of nesting.
        private final Map<Symbol, Symbol> correlationMapping;

        // Apart from immediate correlation and outer correlation, output symbols of left join source are not allowed in right source.
        // However, it is possible to reuse a symbol.
        // forbiddenSymbols is a set of all symbols which mustn't be reused in the given context, coming from all levels of nested joins.
        // If a conflicting symbol is detected, it should be remapped to a new symbol.
        private final Set<Symbol> forbiddenSymbols;

        public UnaliasContext(Map<Symbol, Symbol> correlationMapping, Set<Symbol> forbiddenSymbols)
        {
            this.correlationMapping = requireNonNull(correlationMapping, "correlationMapping is null");
            this.forbiddenSymbols = requireNonNull(forbiddenSymbols, "forbiddenSymbols is null");
        }

        public static UnaliasContext empty()
        {
            return new UnaliasContext(ImmutableMap.of(), ImmutableSet.of());
        }

        public Map<Symbol, Symbol> getCorrelationMapping()
        {
            return correlationMapping;
        }

        public Set<Symbol> getForbiddenSymbols()
        {
            return forbiddenSymbols;
        }
    }

    private static class PlanAndMappings
    {
        private final PlanNode root;
        private final Map<Symbol, Symbol> symbolMappings;

        public PlanAndMappings(PlanNode root, Map<Symbol, Symbol> symbolMappings)
        {
            this.root = requireNonNull(root, "root is null");
            this.symbolMappings = ImmutableMap.copyOf(requireNonNull(symbolMappings, "symbolMappings is null"));
        }

        public PlanNode getRoot()
        {
            return root;
        }

        public Map<Symbol, Symbol> getSymbolMappings()
        {
            return symbolMappings;
        }
    }
}
