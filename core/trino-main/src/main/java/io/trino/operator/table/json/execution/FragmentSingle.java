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
package io.trino.operator.table.json.execution;

import com.fasterxml.jackson.databind.JsonNode;
import com.google.common.collect.ImmutableList;
import io.trino.json.JsonPathEvaluator;
import io.trino.json.ir.IrJsonPath;
import io.trino.json.ir.TypedValue;
import io.trino.operator.table.json.JsonTableColumn;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static io.trino.json.ir.SqlJsonLiteralConverter.getJsonNode;
import static java.util.Objects.requireNonNull;

public class FragmentSingle
        implements JsonTableProcessingFragment
{
    private static final Object[] NO_PARAMETERS = new Object[] {};

    private final JsonPathEvaluator pathEvaluator;
    private final List<Column> columns;
    private final boolean outer;
    private final JsonTableProcessingFragment child;

    // store values produced by columns to reuse them while iterating over child
    // the array has the capacity of all columns produced by json_table
    // the relevant positions can be found by referring to `getOutputIndex()` of each column
    private final Object[] currentValues;
    private final int[] outputLayout;

    // the place where the computed values (or nulls) are stored while computing an output row
    private final Object[] newRow;

    List<JsonNode> sequence;
    int currentItemIndex;

    // start processing next item from the sequence
    boolean processNextItem;

    // do we need to produce null-padded row for OUTER
    boolean childAlreadyProduced;

    public FragmentSingle(IrJsonPath path, List<Column> columns, boolean outer, JsonTableProcessingFragment child, int allColumnsCount, Object[] newRow)
    {
        requireNonNull(path, "path is null");
        //this.pathEvaluator = new JsonPathEvaluator(path, session, metadata, typeManager, functionManager);
        this.pathEvaluator = null;
        this.columns = ImmutableList.copyOf(columns);
        this.outer = outer;
        this.child = requireNonNull(child, "child is null");
        this.currentValues = new Object[allColumnsCount];
        this.outputLayout = IntStream.concat(
                        columns.stream()
                                .mapToInt(Column::getOutputIndex),
                        Arrays.stream(child.getOutputLayout()))
                .toArray();
        this.newRow = requireNonNull(newRow, "newRow is null");
    }

    @Override
    public void reset(JsonNode item)
    {
        reset(item, NO_PARAMETERS);
    }

    /**
     * FragmentSingle can be the root Fragment. The root fragment is the only fragment that may have path parameters.
     * Prepares the root Fragment to produce rows for the new JSON item and a set of path parameters.
     */
    public void reset(JsonNode item, Object[] pathParameters)
    {
        this.sequence = pathEvaluator.evaluate(item, pathParameters).stream()// TODO handle errors when evaluating path
                .map(object -> {
                    if (object instanceof TypedValue typedValue) {
                        return getJsonNode(typedValue).orElseThrow(); // TODO handle conversion error
                    }
                    return (JsonNode) object;
                })
                .collect(toImmutableList());
        this.currentItemIndex = -1;
        this.processNextItem = true;
    }

    @Override
    public boolean getRow()
    {
        // TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public int[] getOutputLayout()
    {
        return outputLayout;
    }
}
