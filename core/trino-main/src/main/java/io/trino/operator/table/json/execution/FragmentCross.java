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

import java.util.Arrays;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static java.util.Objects.requireNonNull;

public class FragmentCross
        implements JsonTableProcessingFragment
{
    private final List<JsonTableProcessingFragment> siblings;

    // store values produced by siblings to reuse them while iterating over another sibling
    // the array has the capacity of all columns produced by json_table
    // the relevant portions can be found by referring to `getOutputLayout()` of each sibling
    private final Object[] currentValues;
    private final int[] outputLayout;

    // the place where the computed values (or nulls) are stored while computing an output row
    private final Object[] newRow;

    int currentSiblingIndex;
    JsonNode currentItem;

    public FragmentCross(List<JsonTableProcessingFragment> siblings, int allColumnsCount, Object[] newRow)
    {
        this.siblings = ImmutableList.copyOf(siblings);
        checkArgument(siblings.size() >= 2, "less than 2 siblings in Cross node");
        this.currentValues = new Object[allColumnsCount];
        this.outputLayout = siblings.stream()
                .map(JsonTableProcessingFragment::getOutputLayout)
                .flatMapToInt(Arrays::stream)
                .toArray();
        this.newRow = requireNonNull(newRow, "newRow is null");
    }

    @Override
    public void reset(JsonNode item)
    {
        this.currentItem = requireNonNull(item, "item is null");
        this.currentSiblingIndex = -1;
        siblings.stream()
                .forEach(sibling -> sibling.reset(item));
        // no need to clear currentValues. They will be overwritten at the first getRow()
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
