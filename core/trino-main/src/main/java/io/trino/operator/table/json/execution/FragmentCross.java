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
import static com.google.common.base.Preconditions.checkState;
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
        this.newRow = requireNonNull(newRow, "newRow is null"); // TODO instantiated once per partition (there are no partitions bc row semantics) as part of DataProcessor - tableFunctionProvider.getDataProcessor(functionHandle)
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
        if (currentSiblingIndex == -1) {
            // beginning of execution for the JSON item
            for (JsonTableProcessingFragment sibling : siblings) {
                boolean siblingProducedRow = sibling.getRow();
                if (!siblingProducedRow) {
                    // if any sibling is empty, the whole CROSS fragment is empty
                    return false;
                }
                else {
                    recordCurrentValues(sibling); // TODO no need to record. The values are already there in `newRow` ready to be reused.
                }
            }
            currentSiblingIndex = siblings.size() - 1;
            return true;
        }

        while (true) {
            JsonTableProcessingFragment currentSibling = siblings.get(currentSiblingIndex);
            boolean currentSiblingProducedRow = currentSibling.getRow();
            if (currentSiblingProducedRow) {
                recordCurrentValues(currentSibling); // TODO no need to record. The values are already there in `newRow` ready to be reused.
                for (int i = 0; i < currentSiblingIndex; i++) {
                    appendCurrentValues(siblings.get(i)); // TODO no need to append. The values are already there in `newRow`.
                }
                for (int i = currentSiblingIndex + 1; i < siblings.size(); i++) {
                    JsonTableProcessingFragment sibling = siblings.get(i);
                    sibling.reset(currentItem);
                    boolean siblingNonEmpty = sibling.getRow();
                    checkState(siblingNonEmpty, "sibling is empty");
                    recordCurrentValues(sibling); // TODO no need to record. The values are already there in `newRow` ready to be reused.
                }
                currentSiblingIndex = siblings.size() - 1;
                return true;
            }

            // current sibling is finished
            if (currentSiblingIndex == 0) {
                // fragment is finished
                return false;
            }
            currentSiblingIndex--;
        }
    }

    private void recordCurrentValues(JsonTableProcessingFragment sibling)
    {
        for (int column : sibling.getOutputLayout()) {
            currentValues[column] = newRow[column];
        }
    }

    private void appendCurrentValues(JsonTableProcessingFragment sibling)
    {
        for (int column : sibling.getOutputLayout()) {
            newRow[column] = currentValues[column];
        }
    }

    @Override
    public int[] getOutputLayout()
    {
        return outputLayout;
    }
}
