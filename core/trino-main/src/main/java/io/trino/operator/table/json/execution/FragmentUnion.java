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

public class FragmentUnion
        implements JsonTableProcessingFragment
{
    private final List<JsonTableProcessingFragment> siblings;
    private final int[] outputLayout;

    // the place where the computed values (or nulls) are stored while computing an output row
    private final Object[] newRow;

    int currentSiblingIndex;
    JsonNode currentItem;

    public FragmentUnion(List<JsonTableProcessingFragment> siblings, Object[] newRow)
    {
        this.siblings = ImmutableList.copyOf(siblings);
        checkArgument(siblings.size() >= 2, "less than 2 siblings in Union node");
        this.outputLayout = siblings.stream()
                .map(JsonTableProcessingFragment::getOutputLayout)
                .flatMapToInt(Arrays::stream)
                .toArray();
        this.newRow = requireNonNull(newRow, "newRow is null"); // TODO instantiated once per partition as part of DataProcessor - tableFunctionProvider.getDataProcessor(functionHandle)
    }

    @Override
    public void reset(JsonNode item)
    {
        this.currentItem = requireNonNull(item, "item is null");
        this.currentSiblingIndex = 0;
        siblings.stream()
                .forEach(sibling -> sibling.reset(item));
    }

    @Override
    public boolean getRow()
    {
        while (true) {
            if (currentSiblingIndex >= siblings.size()) {
                // fragment is finished
                return false;
            }

            boolean currentSiblingProducedRow = siblings.get(currentSiblingIndex).getRow();
            if (currentSiblingProducedRow) {
                for (int i = 0; i < currentSiblingIndex; i++) {
                    appendNulls(siblings.get(i)); // TODO no need to append provided that we fill `newRow` in the beginning (reset), and clear after each sibling is finished. The nulls are already there in `newRow`.
                }
                for (int i = currentSiblingIndex + 1; i < siblings.size(); i++) {
                    appendNulls(siblings.get(i)); // TODO no need to append provided that we fill `newRow` in the beginning (reset), and clear after each sibling is finished. The nulls are already there in `newRow`.
                }
                return true;
            }
            // current sibling is finished
            currentSiblingIndex++;
        }
    }

    private void appendNulls(JsonTableProcessingFragment sibling)
    {
        for (int column : sibling.getOutputLayout()) {
            newRow[column] = null;
        }
    }

    @Override
    public int[] getOutputLayout()
    {
        return outputLayout;
    }
}
